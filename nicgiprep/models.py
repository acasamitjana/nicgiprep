from typing import Union, Literal, Optional
import itertools
import pdb
import copy

import numpy as np
import torch
from torch import nn

#######################################
#   Linear Registration/Deformation   #
#######################################


class InstanceRigidModelLOG(nn.Module):
    """Instance-specific rigid registration via log-space parameterisation.

    Jointly estimates per-session rigid transformations by minimising the
    discrepancy between predicted and observed pairwise log-rigid transforms.
    Rotation and translation are stored as Lie-algebra elements and converted
    to 4×4 matrices via the matrix exponential.

    Attributes
    ----------
    N : int
        Number of sessions.
    K : int
        Number of pairwise combinations ``N*(N-1)//2``.
    angle : torch.nn.Parameter
        Lie-algebra rotation vector per session, shape ``(3, N)``.
    translation : torch.nn.Parameter
        Translation vector per session, shape ``(3, N)``.
    """

    def __init__(
        self,
        session_list: list[Union[str, object]],
        reg_weight: float = 0.001,
        cost: Literal["l1", "l2"] = "l1",
        device: str = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        session_list : list
            Session IDs to include (strings or objects with an ``.id`` attribute).
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

        self.session_list = session_list
        self.N = len(session_list)
        self.K = int(self.N * (self.N - 1) / 2)

        # Parameters
        self.angle = torch.nn.Parameter(torch.zeros(3, self.N))
        self.translation = torch.nn.Parameter(torch.zeros(3, self.N))
        self.angle.requires_grad = True
        self.translation.requires_grad = True

    @property
    def matrix(self) -> np.ndarray:
        return self._compute_matrix().cpu().numpy()

    def _compute_matrix(self) -> torch.Tensor:
        """Compute 4×4 rigid transformation matrices from log-space parameters for all sessions available (N)

        Returns
        -------
        torch.Tensor
            Stacked transformation matrices, shape ``(4, 4, N)``.
        """
        T = torch.zeros((4, 4, self.N))
        for n in range(self.N):
            theta = torch.sqrt(
                torch.sum(self.angle[..., n] ** 2)
            )  # torch.sum(torch.abs(self.angle))
            W = torch.zeros((3, 3))
            W[1, 0], W[0, 1] = self.angle[2, n], -self.angle[2, n]
            W[0, 2], W[2, 0] = self.angle[1, n], -self.angle[1, n]
            W[2, 1], W[1, 2] = self.angle[0, n], -self.angle[0, n]
            V = (
                torch.eye(3)
                + (1 - torch.cos(theta)) / (theta**2) * W
                + (theta - torch.sin(theta)) / (theta**3) * torch.matmul(W, W)
            )

            T[:3, :3, n] = (
                torch.eye(3)
                + torch.sin(theta) / theta * W
                + (1 - torch.cos(theta)) / (theta**2) * torch.matmul(W, W)
            )
            T[:3, 3, n] = (
                V @ self.translation[..., n]
            )  # torch.matmul(V, self.translation[..., n])
            T[3, 3, n] = 1

        return T

    def _build_combinations(
        self, session_list: list[Union[object, str]]
    ) -> torch.Tensor:
        """Build pairwise log-rigid difference vectors for all sessions pairs.
        Total number of registrations is K

        Parameters
        ----------
        session_list : list
            Session IDs to include (same format as passed to ``__init__``).

        Returns
        -------
        torch.Tensor
            Pairwise log-rigid vectors, shape ``(6, K)``.
        """
        K = self.K
        if any([isinstance(t, str) for t in session_list]):
            sessions_dict = {t: it_t for it_t, t in enumerate(session_list)}
        else:
            sessions_dict = {
                t.id: it_t for it_t, t in enumerate(session_list)
            }  # needed for non-consecutive sessions (if we'd like to skip one for whatever reason)

        Tij = torch.zeros((6, K))

        k = 0
        for sess_ref, sess_flo in itertools.combinations(session_list, 2):

            if not isinstance(sess_ref, str):
                t0 = sessions_dict[sess_ref.id]
                t1 = sessions_dict[sess_flo.id]
            else:
                t0 = sessions_dict[sess_ref]
                t1 = sessions_dict[sess_flo]
            Tij[:3, k] = self.angle[..., t1] - self.angle[..., t0]
            Tij[3:, k] = self.translation[..., t1] - self.translation[..., t0]

            k += 1

        return Tij

    def forward(self, logr_obs: torch.Tensor, session_list: list[str]) -> torch.Tensor:
        """Compute the fitting loss between predicted and observed pairwise transforms.

        Parameters
        ----------
        logr_obs : torch.Tensor
            Observed pairwise log-rigid vectors, shape ``(6, K)``.
        session_list : list
            Ordered sessions used to index parameters.

        Returns
        -------
        torch.Tensor
            Scalar loss value.

        Raises
        ------
        ValueError
            If ``self.cost`` is not ``'l1'`` or ``'l2'``.
        """
        logt_ij = self._build_combinations(session_list)
        if self.cost == "l1":
            loss = (
                torch.sum(torch.sqrt(torch.sum((logt_ij - logr_obs) ** 2, axis=0)))
                / self.K
            )
        elif self.cost == "l2":
            loss = torch.sum((logt_ij - logr_obs) ** 2 + 1e-6) / self.K
        else:
            raise ValueError("Cost " + self.cost + " not valid. Choose 'l1' of 'l2'.")
        loss += self.reg_weight * torch.sum(
            torch.sum(self.angle**2, axis=0) + torch.sum(self.translation**2, axis=0),
            axis=0,
        )  # / self.K

        return loss
