import itertools
import pdb
import copy

import numpy as np
import torch
from torch import nn

from nicgiprep.utils import def_utils

#########################################
##   Linear Registration/Deformation   ##
#########################################

class InstanceRigidModelLOG(nn.Module):
    """Instance-specific rigid registration via log-space parameterisation.

    Jointly estimates per-timepoint rigid transformations by minimising the
    discrepancy between predicted and observed pairwise log-rigid transforms.
    Rotation and translation are stored as Lie-algebra elements and converted
    to 4×4 matrices via the matrix exponential.

    Attributes
    ----------
    N : int
        Number of timepoints.
    K : int
        Number of pairwise combinations ``N*(N-1)//2``.
    angle : torch.nn.Parameter
        Lie-algebra rotation vector per timepoint, shape ``(3, N)``.
    translation : torch.nn.Parameter
        Translation vector per timepoint, shape ``(3, N)``.
    """

    def __init__(self, timepoints, reg_weight=0.001, cost='l1', device='cpu', torch_dtype=torch.float):
        """
        Parameters
        ----------
        timepoints : list
            Ordered list of timepoint identifiers (strings or objects with
            an ``.id`` attribute).
        reg_weight : float, optional
            Weight for the L2 regularisation on rotation and translation
            parameters. Default is 0.001.
        cost : {'l1', 'l2'}, optional
            Pairwise fitting loss. Default is ``'l1'``.
        device : str, optional
            PyTorch device string. Default is ``'cpu'``.
        torch_dtype : torch.dtype, optional
            Floating-point dtype for parameters. Default is
            ``torch.float``.
        """
        super().__init__()

        self.device = device
        self.cost = cost
        self.reg_weight = reg_weight

        self.timepoints = timepoints
        self.N = len(timepoints)
        self.K = int(self.N * (self.N-1) / 2)

        # Parameters
        self.angle = torch.nn.Parameter(torch.zeros(3, self.N))
        self.translation = torch.nn.Parameter(torch.zeros(3, self.N))
        self.angle.requires_grad = True
        self.translation.requires_grad = True


    def _compute_matrix(self):
        """Compute 4×4 rigid transformation matrices from log-space parameters.

        Returns
        -------
        torch.Tensor
            Stacked transformation matrices, shape ``(4, 4, N)``.
        """
        T = torch.zeros((4,4,self.N))
        for n in range(self.N):
            theta = torch.sqrt(torch.sum(self.angle[..., n]**2)) # torch.sum(torch.abs(self.angle))
            W = torch.zeros((3,3))
            W[1,0], W[0,1] = self.angle[2, n], -self.angle[2, n]
            W[0,2], W[2,0] = self.angle[1, n], -self.angle[1, n]
            W[2,1], W[1,2] = self.angle[0, n], -self.angle[0, n]
            V = torch.eye(3) + (1 - torch.cos(theta)) / (theta ** 2) * W + (theta - torch.sin(theta)) / (theta ** 3) * torch.matmul(W,W)

            T[:3, :3, n] = torch.eye(3) + torch.sin(theta) / theta * W      +      (1 - torch.cos(theta)) / (theta ** 2) * torch.matmul(W,W)
            T[:3, 3, n] = V @ self.translation[..., n]#torch.matmul(V, self.translation[..., n])
            T[3, 3, n] = 1

            #
            # for n in range(self.N):
            #
            #     T[..., n] = torch.chain_matmul(self.T0inv, T[..., n], self.T0)

        return T


    def _build_combinations(self, timepoints):
        """Build pairwise log-rigid difference vectors for all timepoint pairs.

        Parameters
        ----------
        timepoints : list
            Ordered timepoints (same format as passed to ``__init__``).

        Returns
        -------
        torch.Tensor
            Pairwise log-rigid vectors, shape ``(6, K)``.
        """
        K = self.K
        if any([isinstance(t, str) for t in timepoints]):
            timepoints_dict = {
                t: it_t for it_t, t in enumerate(timepoints)
            }
        else:
            timepoints_dict = {
                t.id: it_t for it_t, t in enumerate(timepoints)
            }  # needed for non consecutive timepoints (if we'd like to skip one for whatever reason)

        Tij = torch.zeros((6, K))

        k = 0
        for tp_ref, tp_flo in itertools.combinations(timepoints, 2):

            if not isinstance(tp_ref, str):
                t0 = timepoints_dict[tp_ref.id]
                t1 = timepoints_dict[tp_flo.id]
            else:
                t0 = timepoints_dict[tp_ref]
                t1 = timepoints_dict[tp_flo]
            Tij[:3, k] = self.angle[..., t1] - self.angle[..., t0]
            Tij[3:, k] = self.translation[..., t1] - self.translation[..., t0]

            k += 1

        return Tij


    def forward(self, logRobs, timepoints):
        """Compute the fitting loss between predicted and observed pairwise transforms.

        Parameters
        ----------
        logRobs : torch.Tensor
            Observed pairwise log-rigid vectors, shape ``(6, K)``.
        timepoints : list
            Ordered timepoints used to index parameters.

        Returns
        -------
        torch.Tensor
            Scalar loss value.

        Raises
        ------
        ValueError
            If ``self.cost`` is not ``'l1'`` or ``'l2'``.
        """
        logTij = self._build_combinations(timepoints)
        if self.cost == 'l1':
            loss = torch.sum(torch.sqrt(torch.sum((logTij - logRobs) ** 2, axis=0))) / self.K
        elif self.cost == 'l2':
            loss = torch.sum((logTij - logRobs) ** 2 + 1e-6) / self.K
        else:
            raise ValueError('Cost ' + self.cost + ' not valid. Choose \'l1\' of \'l2\'.' )
        loss += self.reg_weight * torch.sum(torch.sum(self.angle**2, axis=0) + torch.sum(self.translation**2, axis=0), axis=0) # / self.K

        return loss

class ST2Nonlinear(nn.Module):
    """Spatio-temporal squared (ST²) nonlinear registration model.

    Estimates per-timepoint stationary velocity fields (SVFs) on a
    control-point grid that minimise discrepancy between predicted and
    observed pairwise deformation fields.

    Attributes
    ----------
    N : int
        Number of timepoints.
    K : int
        Number of pairwise combinations ``N*(N-1)//2``.
    T : torch.nn.ParameterDict
        Per-timepoint SVFs keyed by timepoint ID, each of shape
        ``(1, 3, *cp_size)``.
    """

    def __init__(self, obs_size, cp_size, factor=2, cost='l1', timepoints=None, init_T=None, reg_weight=1,
                 version=0, device='cpu'):
        """
        Parameters
        ----------
        obs_size : tuple of int
            Spatial shape of the observation (displacement) grid
            ``(X, Y, Z)``.
        cp_size : tuple of int
            Spatial shape of the control-point grid ``(X', Y', Z')``.
        factor : int, optional
            Up-scale factor from control-point to observation grid.
            Default is 2.
        cost : {'l1', 'l2'}, optional
            Residual fitting loss. Default is ``'l1'``.
        timepoints : list, optional
            Ordered timepoint objects (used when ``init_T`` is ``None``).
        init_T : dict, optional
            Mapping ``{timepoint_id: torch.Tensor}`` to initialise SVFs.
            If provided, ``timepoints`` is ignored.
        reg_weight : float, optional
            Regularisation weight. Default is 1.
        version : int, optional
            Algorithm variant: ``0`` uses ``_build_combinations``;
            ``1`` uses ``_get_difference``. Default is 0.
        device : str, optional
            PyTorch device string. Default is ``'cpu'``.
        """
        super().__init__()
        self.obs_size = obs_size
        self.cost = cost
        self.device = device
        self.factor = factor
        self.reg_weight = reg_weight
        self.version = version

        if init_T is not None:
            self.N = len(init_T)
            self.T = torch.nn.ParameterDict({tid: torch.nn.Parameter(T) for tid, T in init_T.items()}).to(device)

        else:
            self.N = len(timepoints)
            self.T = torch.nn.ParameterDict({t.id: torch.nn.Parameter(torch.zeros((1, 3) + cp_size)) for t in timepoints}).to(device)

        # for T in self.T.values():
        self.T.requires_grad = True

        self.K = int(self.N * (self.N - 1) / 2)

        # ii = torch.arange(0, obs_size[0], dtype=torch.int, device=device)
        # jj = torch.arange(0, obs_size[1], dtype=torch.int, device=device)
        # kk = torch.arange(0, obs_size[2], dtype=torch.int, device=device)
        ii = torch.arange(0, cp_size[0], dtype=torch.int, device=device)
        jj = torch.arange(0, cp_size[1], dtype=torch.int, device=device)
        kk = torch.arange(0, cp_size[2], dtype=torch.int, device=device)
        self.grid = torch.unsqueeze(torch.stack(torch.meshgrid(ii, jj, kk, indexing='ij'), axis=0), 0)

        self.integrate = def_utils.VecInt(cp_size, int_steps=7).to(device)
        self.upscale = def_utils.RescaleTransform(cp_size, factor=self.factor).to(device) if self.factor != 1 else lambda x: x
        self.downscale = def_utils.RescaleTransform(cp_size, factor=1/self.factor).to(device) if self.factor != 1 else lambda x: x
        self.interp = def_utils.SpatialInterpolation().to(device)

    def _compose_fields(self, f1, f2):
        """Compose two displacement fields via spatial interpolation.

        Parameters
        ----------
        f1 : torch.Tensor
            First displacement field, shape ``(1, 3, *cp_size)``.
        f2 : torch.Tensor
            Second displacement field, shape ``(1, 3, *cp_size)``.

        Returns
        -------
        torch.Tensor
            Composed displacement ``f1 ∘ f2``, same shape as ``f1``.
        """
        GG2 = self.grid + f1
        f2_int = self.interp(f2, GG2.clone())
        GG3 = GG2 + f2_int
        return GG3 -  self.grid

    def _build_combinations(self, tid_list):
        """Predict pairwise composed deformations from per-timepoint SVFs.

        Parameters
        ----------
        tid_list : list of str
            Pairs in ``'ref_to_flo'`` format specifying which combinations
            to evaluate.

        Returns
        -------
        torch.Tensor
            Predicted pairwise fields,
            shape ``(*obs_size, 3, len(tid_list))``.
        """
        K = self.K
        k = 0
        R_hat = []#torch.zeros(self.obs_size + (3, K), device=self.device, requires_grad=True)
        for tp_id in tid_list:
            tp_ref, tp_flo = tp_id.split('_to_')
            # FIELD_REF = self.upscale(self.integrate(-self.T[tp_ref]))
            # FIELD_FLO = self.upscale(self.integrate(self.T[tp_flo]))
            FIELD_REF = self.integrate(-self.T[tp_ref])
            FIELD_FLO = self.integrate(self.T[tp_flo])
            R = self._compose_fields(FIELD_REF, FIELD_FLO)
            R_hat += [R]

            k += 1

        R_hat = torch.permute(torch.cat(R_hat, 0), (2, 3, 4, 1, 0))
        return R_hat

    def _get_difference(self, R, tid_list):
        """Compute residuals between observed and predicted pairwise fields.

        Parameters
        ----------
        R : torch.Tensor
            Observed pairwise displacement fields,
            shape ``(*obs_size, 3, K)``.
        tid_list : list of str
            Pairs in ``'ref_to_flo'`` format.

        Returns
        -------
        torch.Tensor
            Residual fields, shape ``(*obs_size, 3, K)``.
        """
        R = torch.permute(R, (4, 3, 0, 1, 2))
        R_hat = []
        for it_tp, tp_id in enumerate(tid_list):
            tp_ref, tp_flo = tp_id.split('_to_')
            FIELD_FLO = self.upscale(self.integrate(-self.T[tp_flo])).float()
            FIELD_REF = self.upscale(self.integrate(self.T[tp_ref])).float()
            GG = self.grid + R[it_tp]
            f1_int = self.interp(FIELD_FLO, GG.clone())
            GG2 = GG + f1_int
            f2_int = self.interp(FIELD_REF, GG2.clone())
            GG3 = GG2 + f2_int
            R_hat += [GG3 -  self.grid]

        R_hat = torch.permute(torch.cat(R_hat, 0), (2, 3, 4, 1, 0))
        return R_hat

    def forward(self, R, tid_list, M=None):
        """Compute the fitting loss between predicted and observed pairwise fields.

        Parameters
        ----------
        R : torch.Tensor
            Observed pairwise displacement fields,
            shape ``(*obs_size, 3, K)``.
        tid_list : list of str
            Pairs in ``'ref_to_flo'`` format.
        M : torch.Tensor, optional
            Spatial weight mask applied to the residuals. If ``None``,
            all voxels contribute equally.

        Returns
        -------
        torch.Tensor
            Scalar loss value.

        Raises
        ------
        ValueError
            If ``self.cost`` is not ``'l1'`` or ``'l2'``.
        """
        if self.version == 0:
            R_hat = self._build_combinations(tid_list)
            residue = R_hat - R

        elif self.version == 1:
            residue = self._get_difference(R, tid_list)

        else:
            R_hat = self._build_combinations(tid_list)
            residue = R_hat - R

        # pdb.set_trace()
        # import nibabel as nib
        # img = nib.Nifti1Image(R_hat[..., 10].cpu().detach().numpy(), np.eye(4))
        # nib.save(img, 'R_hat.prova.nii.gz')
        # img = nib.Nifti1Image(R[..., 10].cpu().detach().numpy(), np.eye(4))
        # nib.save(img, 'R.prova.nii.gz')

        if M is None:
            reduce_fn = torch.sum
        else:
            reduce_fn = lambda x: torch.sum(x*M)#/torch.sum(M)

        if self.cost == 'l1':
            loss = reduce_fn(torch.sqrt(torch.sum((residue) ** 2, axis=-2)))
        elif self.cost == 'l2':
            loss = reduce_fn(torch.sum((residue) ** 2, axis=-2))
        else:
            raise ValueError('Cost ' + self.cost + ' not valid. Choose \'l1\' of \'l2\'.' )

        # reg_loss = torch.sum(torch.sqrt(torch.sum(torch.sum(torch.cat([T for T in self.T.values()], dim=0), dim=0)**2, dim=0)))
        # loss += self.reg_weight * reg_loss

        # loss.backward()
        # for p in self.parameters():
        #     print(torch.max(torch.sqrt(torch.sum(p.grad ** 2, dim=1))))

        return loss
