from os.path import join
import os
import csv

import numpy as np
import torch



class Callback(object):
    """Base callback with no-op implementations of all training hooks.

    Subclass and override the relevant methods to inject behaviour at
    specific points in the training loop.
    """

    def on_train_init(self, model, **kwargs):
        """Called once before training begins."""
        pass

    def on_train_fi(self, model, **kwargs):
        """Called once after training finishes."""
        pass

    def on_epoch_init(self, model, epoch, **kwargs):
        """Called at the start of each epoch.

        Parameters
        ----------
        model : nn.Module
            The model being trained.
        epoch : int
            Current epoch index.
        """
        pass

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        """Called at the end of each epoch.

        Parameters
        ----------
        logs_dict : dict
            Validation metrics for the finished epoch.
        model : nn.Module
            The model being trained.
        epoch : int
            Current epoch index.
        """
        pass

    def on_step_init(self, logs_dict, model, epoch, **kwargs):
        """Called at the start of each optimisation step."""
        pass

    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        """Called at the end of each optimisation step.

        Parameters
        ----------
        logs_dict : dict
            Metrics produced during the step.
        model : nn.Module
            The model being trained.
        epoch : int
            Current epoch index.
        """
        pass

class History(Callback):
    """Accumulate training and validation metrics in memory.

    Attributes
    ----------
    logs : dict
        Nested dictionary ``{'Train': {epoch: {key: [values]}},
        'Validation': {epoch: {key: value}}}``.
    keys : list of str
        Metric names to track per training step.
    """

    def __init__(self, keys=None):
        """
        Parameters
        ----------
        keys : list of str, optional
            Metric keys to initialise in the per-step log. Defaults to
            an empty list.
        """
        self.logs = {}

        if keys is None:
            self.keys = []
        else:
            self.keys = keys

    def on_train_init(self, model, **kwargs):
        """Initialise the Train and Validation log containers."""
        self.logs['Train'] = {}
        self.logs['Validation'] = {}


    def on_epoch_init(self, model, epoch, **kwargs):
        """Create per-epoch sub-dicts and initialise tracked key lists."""
        self.logs['Train'][epoch] = {}
        self.logs['Validation'][epoch] = {}

        for k in self.keys:
            self.logs['Train'][epoch][k] = []


    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        """Append step metrics to the current epoch's training log."""
        for k,v in logs_dict.items():
            self.logs['Train'][epoch][k].append(v)


    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        """Record validation metrics for the finished epoch."""
        for k,v in logs_dict.items():
            self.logs['Validation'][epoch][k]=v

class ModelCheckpoint(Callback):
    """Save model (and optimizer) state at the end of every epoch.

    Always writes ``model_checkpoint.LAST.pth`` and additionally writes
    ``model_checkpoint.<epoch>.pth`` every ``save_model_frequency`` epochs.
    On training completion writes ``model_checkpoint.FI.pth``.

    Attributes
    ----------
    dirpath : str
        Directory where checkpoint files are written.
    save_model_frequency : int
        Epoch interval for periodic snapshots.
    """

    def __init__(self, dirpath, save_model_frequency):
        """
        Parameters
        ----------
        dirpath : str
            Directory where checkpoints are saved.
        save_model_frequency : int
            Save a numbered snapshot every this many epochs.
        """
        self.dirpath = dirpath
        self.save_model_frequency = save_model_frequency

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        """Save ``LAST`` checkpoint and a numbered snapshot every N epochs."""
        optimizer = kwargs['optimizer']
        checkpoint = {
            'epoch': epoch + 1,
        }
        if isinstance(model, dict):
            for model_name, model_instance in model.items():
                checkpoint['state_dict_' + model_name] = model_instance.state_dict()
        else:
            checkpoint['state_dict'] = model.state_dict()

        if isinstance(optimizer, dict):
            for optimizer_name, optimizer_instance in optimizer.items():
                checkpoint['optimizer_' + optimizer_name] = optimizer_instance.state_dict()
        else:
            checkpoint['optimizer'] = optimizer.state_dict()

        filepath = join(self.dirpath, 'model_checkpoint.LAST.pth')
        torch.save(checkpoint, filepath)
        if np.mod(epoch, self.save_model_frequency) == 0:
            filepath = join(self.dirpath, 'model_checkpoint.' + str(epoch) + '.pth')
            torch.save(checkpoint, filepath)


    def on_train_fi(self, model, **kwargs):
        """Save the final ``FI`` checkpoint when training completes."""
        checkpoint = {}
        if isinstance(model, dict):
            for model_name, model_instance in model.items():
                checkpoint['state_dict_' + model_name] = model_instance.state_dict()
        else:
            checkpoint['state_dict'] = model.state_dict()

        filepath = join(self.dirpath, 'model_checkpoint.FI.pth')
        torch.save(checkpoint, filepath)

class PrinterCallback(Callback):
    """Print training progress to stdout at each step and epoch boundary.

    Attributes
    ----------
    keys : list of str or None
        Metric keys to display. If ``None``, all keys in ``logs_dict``
        are printed.
    logs : dict
        Internal log buffer (currently unused but reserved for future use).
    """

    def __init__(self, keys=None):
        """
        Parameters
        ----------
        keys : list of str, optional
            Metric keys to print. Defaults to ``None`` (print all).
        """
        self.keys = keys
        self.logs = {}

    def on_train_init(self, model, **kwargs):
        """Print the training-start banner."""
        print('\n')
        print('\n')
        print('######################################')
        print('########## Training started ##########')
        print('######################################')
        print('\n')

    def on_epoch_init(self, model, epoch, **kwargs):
        """Print the current epoch number."""
        print('------------- Epoch: ' + str(epoch))

    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        """Print iteration index and metric values for the finished step."""
        to_print = 'Iteration: (' + str(kwargs['iteration']) + '/' + str(kwargs['N']) + '). ' + \
                   ', '.join([k + ': ' + str(round(v, 6)) for k, v in logs_dict.items()])
        print(to_print)

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        """Print a summary of epoch-end validation metrics."""
        to_print = 'Epoch summary: ' + ','.join([k + ': ' + str(round(v, 6)) for k, v in logs_dict.items()])
        print(to_print)

    def on_train_fi(self, model, **kwargs):
        """Print the training-finished banner."""
        print('#######################################')
        print('########## Training finished ##########')
        print('#######################################')

