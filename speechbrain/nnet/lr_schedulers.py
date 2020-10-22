"""
Learning rate schedulers.

Authors
 * Mirco Ravanelli 2020
"""

import math
import torch
import logging
from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)


@checkpoints.register_checkpoint_hooks
class NewBobLRScheduler:
    """Learning rate scheduler with new-bob technique.

     The learning rate is annealed based on the validation peformance.
     In particular: if (past_loss-current_loss)/past_loss< impr_threshold:
     lr=lr * annealing_factor

    Arguments
    ---------
    annealing_factor : float
        It is annealing factor used in new_bob strategy.
    improvement_threshold : float
        It is improvement rate between losses used to perform learning
        annealing in new_bob strategy.
    patient : int
        When the annealing condition is violeted patient times,
        the learning rate is finally reduced.

    Example
    -------
    >>> from speechbrain.nnet.optimizers import SGD_Optimizer
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> optim = SGD_Optimizer(learning_rate=1.0)
    >>> output = model(inp_tensor, init_params=True)
    >>> optim.init_params([model])
    >>> scheduler = NewBobLRScheduler()
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=1, current_loss=10.0)
    >>> optim.optim.param_groups[0]["lr"]
    1.0
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=2, current_loss=2.0)
    >>> optim.optim.param_groups[0]["lr"]
    1.0
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=3, current_loss=2.5)
    >>> optim.optim.param_groups[0]["lr"]
    0.5
    """

    def __init__(
        self, annealing_factor=0.5, improvement_threshold=0.0025, patient=0,
    ):
        self.annealing_factor = annealing_factor
        self.improvement_threshold = improvement_threshold
        self.patient = patient
        self.losses = []
        self.current_patient = self.patient

    def __call__(self, optim_list, current_epoch, current_loss):
        """
        Arguments
        ---------
        optim_list : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.

        Returns
        -------
        float
            The learning rate before the update.
        float
            The learning rate after the update.
        """
        for opt in optim_list:
            current_lr = opt.optim.param_groups[0]["lr"]

            # Learning rate annealing
            next_lr = current_lr
            if len(self.losses) > 0:
                last_loss = self.losses[-1]
                next_lr = self._new_bob_scheduler(
                    last_loss, current_loss, current_lr
                )

            # Changing the learning rate within the optimizer
            opt.optim.param_groups[0]["lr"] = next_lr
            opt.optim.param_groups[0]["prev_lr"] = current_lr
            if next_lr != current_lr:
                logger.info(
                    "Changing lr from %.2g to %.2g" % (current_lr, next_lr)
                )
        # Updating current loss
        self.losses.append(current_loss)

        return current_lr, next_lr

    def _new_bob_scheduler(self, last_loss, current_loss, current_lr):
        """
        Arguments
        ---------
        last_loss : float
            Loss observed in the previous epoch.
        current_loss : int
            Loss of the current epoch.
        current_lr : float
            Learning rate used to process the current epoch.

        Returns
        -------
        next_lr: float
            The learning rate to use for the next epoch
        """
        next_lr = current_lr
        current_improvement = (last_loss - current_loss) / abs(last_loss)
        if current_improvement < self.improvement_threshold:
            if self.current_patient == 0:
                next_lr = current_lr * self.annealing_factor
                self.current_patient = self.patient
            else:
                self.current_patient = self.current_patient - 1
        return next_lr

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses, "current_patient": self.current_patient}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch):
        del end_of_epoch  # Unused in this class
        data = torch.load(path)
        self.losses = data["losses"]
        self.current_patient = data["current_patient"]


@checkpoints.register_checkpoint_hooks
class LinearLRScheduler:
    """Learning rate scheduler with linear annealing technique.

     The learning rate linearly decays over the epochs between lr_initial
     and lr_final.

    Arguments
    ---------
    lr_initial : float
        Initial learning rate (i.e. the lr used at epoch 0).
    lr_final : float
        Learning rate to use in the final epoch.
    N_epochs : int
        Number of epochs.

    Example
    -------
    >>> from speechbrain.nnet.optimizers import SGD_Optimizer
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> optim = SGD_Optimizer(learning_rate=1.0)
    >>> output = model(inp_tensor, init_params=True)
    >>> optim.init_params([model])
    >>> scheduler = LinearLRScheduler(optim.optim.param_groups[0]["lr"], 0.0, 4)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=1, current_loss=10.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=2, current_loss=10.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=3, current_loss=10.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=4, current_loss=10.0)
    >>> optim.optim.param_groups[0]["lr"]
    0.0
    """

    def __init__(
        self, lr_initial, lr_final, N_epochs,
    ):
        self.lr_initial = lr_initial
        self.lr_final = lr_final
        self.N_epochs = N_epochs
        self.losses = []

        # LR planning
        self.lr_at_epoch = torch.linspace(
            self.lr_initial, self.lr_final, steps=self.N_epochs
        ).tolist()

    def __call__(self, optim_list, current_epoch, current_loss):
        """
        Arguments
        ---------
        optim_list : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.

        Returns
        -------
        float
            The learning rate before the update.
        float
            The learning rate after the update.
        """
        for opt in optim_list:
            current_lr = opt.optim.param_groups[0]["lr"]
            if current_epoch < len(self.lr_at_epoch):
                next_lr = self.lr_at_epoch[current_epoch]
            else:
                next_lr = current_lr

            # Changing the learning rate within the optimizer
            opt.optim.param_groups[0]["lr"] = next_lr
            opt.optim.param_groups[0]["prev_lr"] = current_lr
            if next_lr != current_lr:
                logger.info(
                    "Changing lr from %.2g to %.2g" % (current_lr, next_lr)
                )
        # Updating current loss
        self.losses.append(current_loss)

        return current_lr, next_lr

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch):
        del end_of_epoch  # Unused in this class
        data = torch.load(path)
        self.losses = data["losses"]


@checkpoints.register_checkpoint_hooks
class StepLRScheduler:
    """Learning rate scheduler with step annealing technique.

     The leatning rate decays over the epochs with the selected epoch_decay
     factor lr=self.lr_int*epoch_decay^((1+epoch)/self.epoch_drop)

    Arguments
    ---------
    lr_initial : float
        Initial learning rate (i.e. the lr used at epoch 0).
    epoch_decay : float
        Decay factor used in time and step decay strategies.
    epoch_drop: float
        Annealing factor (the decay of the lr rate is faster with higher
        values).

    Example
    -------
    >>> from speechbrain.nnet.optimizers import SGD_Optimizer
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> optim = SGD_Optimizer(learning_rate=1.0)
    >>> output = model(inp_tensor, init_params=True)
    >>> optim.init_params([model])
    >>> scheduler =StepLRScheduler(optim.optim.param_groups[0]["lr"])
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=1, current_loss=10.0)
    >>> optim.optim.param_groups[0]["lr"]
    0.5
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=2, current_loss=2.0)
    >>> optim.optim.param_groups[0]["lr"]
    0.5
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=3, current_loss=2.5)
    >>> optim.optim.param_groups[0]["lr"]
    0.25
    """

    def __init__(
        self, lr_initial, epoch_decay=0.5, epoch_drop=2,
    ):
        self.lr_initial = lr_initial
        self.epoch_decay = epoch_decay
        self.epoch_drop = epoch_drop
        self.losses = []

    def __call__(self, optim_list, current_epoch, current_loss):
        """
        Arguments
        ---------
        optim_list : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.

        Returns
        -------
        float
            The learning rate before the update.
        float
            The learning rate after the update.
        """
        for opt in optim_list:
            current_lr = opt.optim.param_groups[0]["lr"]
            next_lr = self.lr_initial * math.pow(
                self.epoch_decay,
                math.floor((1 + current_epoch) / self.epoch_drop),
            )

            # Changing the learning rate within the optimizer
            opt.optim.param_groups[0]["lr"] = next_lr
            opt.optim.param_groups[0]["prev_lr"] = current_lr
            if next_lr != current_lr:
                logger.info(
                    "Changing lr from %.2g to %.2g" % (current_lr, next_lr)
                )
        # Updating current loss
        self.losses.append(current_loss)

        return current_lr, next_lr

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch):
        del end_of_epoch  # Unused in this class
        data = torch.load(path)
        self.losses = data["losses"]


@checkpoints.register_checkpoint_hooks
class NoamScheduler:
    """The is an implementation of the transformer's learning rate scheduler with warmup.
    Reference: https://arxiv.org/abs/1706.03762

    Arguments
    ---------
    lr_initial : float
        Initial learning rate (i.e. the lr used at epoch 0).
    n_warmup_steps : int
        numer of warm up steps

    Example
    -------
    >>> from speechbrain.nnet.optimizers import SGD_Optimizer
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> optim = SGD_Optimizer(learning_rate=1.0)
    >>> output = model(inp_tensor, init_params=True)
    >>> optim.init_params([model])
    >>> scheduler =NoamScheduler(optim.optim.param_groups[0]["lr"], 3)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=1, current_loss=10.0)
    >>> optim.optim.param_groups[0]["lr"]
    0.33333333333333337
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=2, current_loss=2.0)
    >>> optim.optim.param_groups[0]["lr"]
    0.6666666666666667
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=3, current_loss=2.5)
    >>> optim.optim.param_groups[0]["lr"]
    1.0
    """

    def __init__(self, lr_initial, n_warmup_steps, model_size=None):
        self.lr_initial = lr_initial
        self.n_warmup_steps = n_warmup_steps
        self.losses = []
        self.current_lr = lr_initial

        self.n_steps = 0
        self.normalize = 1 / (n_warmup_steps * n_warmup_steps ** -1.5)
        if model_size is not None:
            self.normalize = model_size ** (-0.5)

    def __call__(self, optim_list, current_epoch, current_loss):
        """
        Arguments
        ---------
        optim_list : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.
        Returns
        -------
        float
            The learning rate before the update.
        float
            The learning rate after the update.
        """
        self.n_steps += 1

        for opt in optim_list:
            current_lr = opt.optim.param_groups[0]["lr"]

            lr = self.lr_initial * self._get_lr_scale()

            # Changing the learning rate within the optimizer
            for param_group in opt.optim.param_groups:
                param_group["lr"] = lr

        self.losses.append(current_loss)
        self.current_lr = current_lr
        return current_lr, lr

    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return self.normalize * min(
            n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
        )

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses, "n_steps": self.n_steps}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch):
        del end_of_epoch  # Unused in this class
        data = torch.load(path)
        self.losses = data["losses"]
        self.n_steps = data["n_steps"]


@checkpoints.register_checkpoint_hooks
class CustomLRScheduler:
    """Custom Learning rate scheduler.

     The leatning rate is changed according to a list given by the user.

    Arguments
    ---------
    lr_at_epoch : list
        It is a list of floats containing the learning rates to use for each
        epoch. The length of the list must be equal to the number of epochs.

    Example
    -------
    >>> from speechbrain.nnet.optimizers import SGD_Optimizer
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> optim = SGD_Optimizer(learning_rate=1.0)
    >>> output = model(inp_tensor, init_params=True)
    >>> optim.init_params([model])
    >>> scheduler = CustomLRScheduler(lr_at_epoch=[1.0,0.8,0.6,0.5])
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=1, current_loss=10.0)
    >>> optim.optim.param_groups[0]["lr"]
    0.8
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=2, current_loss=2.0)
    >>> optim.optim.param_groups[0]["lr"]
    0.6
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=3, current_loss=2.5)
    >>> optim.optim.param_groups[0]["lr"]
    0.5
    """

    def __init__(self, lr_at_epoch):
        self.lr_at_epoch = lr_at_epoch
        self.losses = []

    def __call__(self, optim_list, current_epoch, current_loss):
        """
        Arguments
        ---------
        optim_list : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.

        Returns
        -------
        float
            The learning rate before the update.
        float
            The learning rate after the update.
        """
        for opt in optim_list:
            current_lr = opt.optim.param_groups[0]["lr"]
            next_lr = self.lr_at_epoch[current_epoch]

            # Changing the learning rate within the optimizer
            opt.optim.param_groups[0]["lr"] = next_lr
            opt.optim.param_groups[0]["prev_lr"] = current_lr
            if next_lr != current_lr:
                logger.info(
                    "Changing lr from %.2g to %.2g" % (current_lr, next_lr)
                )
        # Updating current loss
        self.losses.append(current_loss)

        return current_lr, next_lr

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch):
        del end_of_epoch  # Unused in this class
        data = torch.load(path)
        self.losses = data["losses"]


@checkpoints.register_checkpoint_hooks
class ReduceLROnPlateau:
    """Learning rate scheduler with linear annealing technique.

     The learning rate linearly decays over the epochs between lr_initial
     and lr_final.

    Arguments
    ---------
    lr_min: float
        The minimum allowable learning rate
    factor: float
        Factor with which to reduce the learning rate
    patience: int
        How many

    Example
    -------
    >>> from speechbrain.nnet.optimizers import SGD_Optimizer
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> optim = SGD_Optimizer(learning_rate=1.0)
    >>> output = model(inp_tensor, init_params=True)
    >>> optim.init_params([model])
    >>> scheduler = LinearLRScheduler(optim.optim.param_groups[0]["lr"], 0.0, 4)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=1, current_loss=10.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=2, current_loss=10.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=3, current_loss=10.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=4, current_loss=10.0)
    >>> optim.optim.param_groups[0]["lr"]
    0.0
    """

    def __init__(
        self, lr_min=1e-8, factor=0.5, patience=2,
    ):
        self.lr_min = lr_min
        self.factor = factor
        self.patience = patience
        self.patience_counter = 0
        self.losses = []

    def __call__(self, optim_list, current_epoch, current_loss):
        """
        Arguments
        ---------
        optim_list : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.

        Returns
        -------
        float
            The learning rate before the update.
        float
            The learning rate after the update.
        """
        for opt in optim_list:
            current_lr = opt.optim.param_groups[0]["lr"]

            # last_p_epochs = self.loses[-self.patience:]
            if current_epoch == 0:
                next_lr = current_lr
                self.anchor = current_loss
            else:
                if current_loss < self.anchor:
                    self.patience_counter = 0
                    next_lr = current_lr
                    self.anchor = current_loss
                elif (
                    current_loss > self.anchor
                    and self.patience_counter < self.patience
                ):
                    self.patience_counter = self.patience_counter + 1
                    next_lr = current_lr
                else:
                    next_lr = current_lr * self.factor

            # impose the lower bound
            next_lr = max(next_lr, self.lr_min)

            # Changing the learning rate within the optimizer
            opt.optim.param_groups[0]["lr"] = next_lr
            opt.optim.param_groups[0]["prev_lr"] = current_lr
            if next_lr != current_lr:
                logger.info(
                    "Changing lr from %.2g to %.2g" % (current_lr, next_lr)
                )
        # Updating current loss
        self.losses.append(current_loss)

        return current_lr, next_lr

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch):
        del end_of_epoch  # Unused in this class
        data = torch.load(path)
        self.losses = data["losses"]
