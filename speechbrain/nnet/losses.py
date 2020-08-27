"""
Losses for training neural networks.

Authors
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Hwidong Na 2020
"""

import torch
from torch import nn
import torch.nn.functional as F
import logging
import functools
from speechbrain.data_io.data_io import length_to_mask
from itertools import permutations


logger = logging.getLogger(__name__)


def transducer_loss(log_probs, targets, input_lens, target_lens, blank_index):
    """Transducer loss, see `speechbrain/nnet/transducer/transducer_loss.py`

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape [batch, time, chars].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len]
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the character indexes.
    """
    from speechbrain.nnet.transducer.transducer_loss import Transducer

    input_lens = (input_lens * log_probs.shape[1]).int()
    target_lens = (target_lens * targets.shape[1]).int()
    return Transducer.apply(
        log_probs,
        targets,
        input_lens,
        target_lens,
        blank_index,
        reduction="mean",
    )


class PitWrapper(nn.Module):
    """
    Permutation Invariant Wrapper to allow Permutation Invariant Training
    (PIT) with existing losses.
    Permutation invariance is calculated over sources/classes axis which is
    assumed to be the rightmost dimension: predictions and targets tensors are
    assumed to have shape [batch, ..., channels, sources].

    Arguments
    ---------
    base_loss : function
        base loss function, e.g. torch.nn.MSELoss. It is assumed that it takes
        two arguments:
        predictions and targets and no reduction is performed.
        (if a pytorch loss is used, the user must specify reduction="none").

    Returns
    ---------
    pit_loss : torch.nn.Module
        torch module supporting forward method for PIT.

    Example
    -------
    >>> pit_mse = PitWrapper(nn.MSELoss(reduction="none"))
    >>> targets = torch.rand((2, 32, 4))
    >>> p = (3, 0, 2, 1)
    >>> predictions = targets[..., p]
    >>> loss, opt_p = pit_mse(predictions, targets)
    >>> loss
    tensor([0., 0.])
    """

    def __init__(self, base_loss):
        super(PitWrapper, self).__init__()
        self.base_loss = base_loss

    def _fast_pit(self, loss_mat):
        """
        Arguments
        ----------
        loss_mat: torch.Tensor
            tensor of shape [sources, source] containing loss values for each
            possible permutation of predictions.

        Returns
        -------
        loss: torch.Tensor
            permutation invariant loss for current batch, tensor of shape [1]

        assigned_perm: tuple
            indexes for optimal permutation of the input over sources which
            minimizes the loss.
        """

        loss = None
        assigned_perm = None
        for p in permutations(range(loss_mat.shape[0])):
            c_loss = loss_mat[range(loss_mat.shape[0]), p].mean()
            if loss is None or loss > c_loss:
                loss = c_loss
                assigned_perm = p
        return loss, assigned_perm

    def _opt_perm_loss(self, pred, target):
        """

        Parameters
        ----------
        pred: torch.Tensor
            network prediction for current example, tensor of
            shape [..., sources].
        target: torch.Tensor
            target for current example, tensor of shape [..., sources].

        Returns
        -------
        loss: torch.Tensor
            permutation invariant loss for current example, tensor of shape [1]

        assigned_perm: tuple
            indexes for optimal permutation of the input over sources which
            minimizes the loss.

        """

        n_sources = pred.size(-1)
        pred = pred.unsqueeze(-2).repeat(
            *[1 for x in range(len(pred.shape) - 1)], n_sources, 1
        )
        target = target.unsqueeze(-1).repeat(
            1, *[1 for x in range(len(target.shape) - 1)], n_sources
        )

        loss_mat = self.base_loss(pred, target)
        assert (
            len(loss_mat.shape) >= 2
        ), "Base loss should not perform any reduction operation"
        mean_over = [x for x in range(len(loss_mat.shape))]
        loss_mat = loss_mat.mean(dim=mean_over[:-2])

        return self._fast_pit(loss_mat)

    def reorder_tensor(self, tensor, p):
        """
            Arguments
            ---------
            tensor : torch.Tensor
                tensor to reorder given the optimal permutation, of shape
                [batch, ..., sources].
            p : list of tuples
                list of optimal permutations, e.g. for batch=2 and n_sources=3
                [(0, 1, 2), (0, 2, 1].

            Returns
            -------
            reordered: torch.Tensor
                reordered tensor given permutation p.
        """

        reordered = torch.zeros_like(tensor).to(tensor.device)
        for b in range(tensor.shape[0]):
            reordered[b] = tensor[b][..., p[b]].clone()
        return reordered

    def forward(self, preds, targets):
        """
            Arguments
            ---------
            preds : torch.Tensor
                Network predictions tensor, of shape
                [batch, channels, ..., sources].
            targets : torch.Tensor
                Target tensor, of shape [batch, channels, ..., sources].

            Returns
            -------
            loss: torch.Tensor
                permutation invariant loss for current examples, tensor of
                shape [batch]

            perms: list
                list of indexes for optimal permutation of the inputs over
                sources.
                e.g. [(0, 1, 2), (2, 1, 0)] for three sources and 2 examples
                per batch.
        """
        losses = []
        perms = []
        for pred, label in zip(preds, targets):
            loss, p = self._opt_perm_loss(pred, label)
            perms.append(p)
            losses.append(loss)
        loss = torch.stack(losses)
        return loss, perms


def ctc_loss(log_probs, targets, input_lens, target_lens, blank_index):
    """CTC loss

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape [batch, time, chars].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len]
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the character indexes.
    """
    input_lens = (input_lens * log_probs.shape[1]).int()
    target_lens = (target_lens * targets.shape[1]).int()
    log_probs = log_probs.transpose(0, 1)
    return torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lens,
        target_lens,
        blank_index,
        zero_infinity=True,
    )


def l1_loss(predictions, targets, length=None, allowed_len_diff=3):
    """Compute the true l1 loss, accounting for length differences.

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape ``[batch, time, *]``.
    targets : torch.Tensor
        Target tensor, same size as predicted tensor.
    length : torch.Tensor
        Length of each utterance for computing true error with a mask.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1, 0.1, 0.9]])
    >>> l1_loss(probs, torch.tensor([[1., 0., 0., 1.]]))
    tensor(0.1000)
    """
    predictions, targets = truncate(predictions, targets, allowed_len_diff)
    loss = functools.partial(torch.nn.functional.l1_loss, reduction="none")
    return compute_masked_loss(loss, predictions, targets, length)


def mse_loss(predictions, targets, length=None, allowed_len_diff=3):
    """Compute the true mean squared error, accounting for length differences.

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape ``[batch, time, *]``.
    targets : torch.Tensor
        Target tensor, same size as predicted tensor.
    length : torch.Tensor
        Length of each utterance for computing true error with a mask.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1, 0.1, 0.9]])
    >>> mse_loss(probs, torch.tensor([[1., 0., 0., 1.]]))
    tensor(0.0100)
    """
    predictions, targets = truncate(predictions, targets, allowed_len_diff)
    loss = functools.partial(torch.nn.functional.mse_loss, reduction="none")
    return compute_masked_loss(loss, predictions, targets, length)


def classification_error(
    probabilities, targets, length=None, allowed_len_diff=3
):
    """Computes the classification error at frame or batch level.

    Arguments
    ---------
    probabilities : torch.Tensor
        The posterior probabilities of shape
        [batch, prob] or [batch, frames, prob]
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames]
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.

    Example
    -------
    >>> probs = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    >>> classification_error(probs, torch.tensor([1, 1]))
    tensor(0.5000)
    """
    if len(probabilities.shape) == 3 and len(targets.shape) == 2:
        probabilities, targets = truncate(
            probabilities, targets, allowed_len_diff
        )

    def error(predictions, targets):
        predictions = torch.argmax(probabilities, dim=-1)
        return (predictions != targets).float()

    return compute_masked_loss(error, probabilities, targets.long(), length)


def nll_loss(
    log_probabilities,
    targets,
    length=None,
    label_smoothing=0.0,
    allowed_len_diff=3,
):
    """Computes negative log likelihood loss.

    Arguments
    ---------
    log_probabilities : torch.Tensor
        The probabilities after log has been applied.
        Format is [batch, log_p] or [batch, frames, log_p]
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames]
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    >>> nll_loss(torch.log(probs), torch.tensor([1, 1]))
    tensor(1.2040)
    """
    if len(log_probabilities.shape) == 3:
        log_probabilities, targets = truncate(
            log_probabilities, targets, allowed_len_diff
        )
        log_probabilities = log_probabilities.transpose(1, -1)

    # Pass the loss function but apply reduction="none" first
    loss = functools.partial(torch.nn.functional.nll_loss, reduction="none")
    return compute_masked_loss(
        loss,
        log_probabilities,
        targets.long(),
        length,
        label_smoothing=label_smoothing,
    )


def kldiv_loss(
    log_probabilities,
    targets,
    length=None,
    label_smoothing=0.0,
    allowed_len_diff=3,
    pad_idx=0,
):
    """Computes the KL-divergence error at batch level.
    This loss applies label smoothing directly on the targets

    Arguments
    ---------
    probabilities : torch.Tensor
        The posterior probabilities of shape
        [batch, prob] or [batch, frames, prob]
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames]
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    >>> kldiv_loss(torch.log(probs), torch.tensor([1, 1]))
    tensor(1.2040)
    """
    # if the input shape is 2m unsqueeze
    if log_probabilities.dim() == 2:
        log_probabilities = log_probabilities.unsqueeze(1)

    bz, time, n_class = log_probabilities.shape
    targets = targets.long().detach()

    if label_smoothing > 0:
        confidence = 1 - label_smoothing

        true_distribution = torch.nn.functional.one_hot(
            targets, n_class
        ).float()
        true_distribution = true_distribution * confidence + (
            1 - true_distribution
        ) / (n_class - 2)

        # discourage predition of <pad> by setting its corresponding dimention in true dristribution with 0
        true_distribution[:, :, pad_idx] = 0

        loss = functools.partial(torch.nn.functional.kl_div, reduction="none")
        return compute_masked_loss(
            loss, log_probabilities, true_distribution, length
        )
    else:
        log_probabilities = log_probabilities.view(bz, n_class, time)
        targets = targets.view(bz, time)
        loss = functools.partial(
            torch.nn.functional.nll_loss, ignore_index=pad_idx, reduction="none"
        )
        return compute_masked_loss(
            loss, log_probabilities, targets.long(), length
        )


def truncate(predictions, targets, allowed_len_diff=3):
    """Ensure that predictions and targets are the same length.

    Arguments
    ---------
    predictions : torch.Tensor
        First tensor for checking length.
    targets : torch.Tensor
        Second tensor for checking length.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    """
    len_diff = predictions.shape[1] - targets.shape[1]
    if len_diff == 0:
        return predictions, targets
    elif abs(len_diff) > allowed_len_diff:
        raise ValueError(
            "Predictions and targets should be same length, but got %s and "
            "%s respectively." % (predictions.shape[1], targets.shape[1])
        )
    elif len_diff < 0:
        return predictions, targets[:, : predictions.shape[1]]
    else:
        return predictions[:, : targets.shape[1]], targets


def compute_masked_loss(
    loss_fn, predictions, targets, length=None, label_smoothing=0.0
):
    """Compute the true average loss of a set of waveforms of unequal length.

    Arguments
    ---------
    loss_fn : function
        A function for computing the loss taking just predictions and targets.
        Should return all the losses, not a reduction (e.g. reduction="none")
    predictions : torch.Tensor
        First argument to loss function.
    targets : torch.Tensor
        Second argument to loss function.
    length : torch.Tensor
        Length of each utterance to compute mask. If None, global average is
        computed and returned.
    label_smoothing: float
        The proportion of label smoothing. Should only be used for NLL loss.
        Ref: Regularizing Neural Networks by Penalizing Confident Output Distributions.
        https://arxiv.org/abs/1701.06548
    """
    mask = torch.ones_like(targets)
    if length is not None:
        mask = length_to_mask(
            length * targets.shape[1], max_len=targets.shape[1],
        )
        if len(targets.shape) == 3:
            mask = mask.unsqueeze(2).repeat(1, 1, targets.shape[2])

    loss = torch.sum(loss_fn(predictions, targets) * mask) / torch.sum(mask)
    if label_smoothing == 0:
        return loss
    else:
        loss_reg = -torch.sum(
            torch.mean(predictions, dim=1) * mask
        ) / torch.sum(mask)
        return label_smoothing * loss_reg + (1 - label_smoothing) * loss


class Projection(torch.nn.Module):
    """This class implements a projection on the top of network outputs.

    Arguments
    ---------
    inp_neurons : int
        Number of neurons in the network output
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of neurons in final layer

    Example
    -------
    >>> outputs = torch.tensor([ [1., 0.], [0., 1.], [-1., 0.], [0., -1.] ])
    >>> projection = Projection(inp_neurons=2, out_neurons=1)
    >>> logits = projection(outputs)
    >>> logits.shape
    torch.Size([4, 1])
    """

    def __init__(
        self,
        inp_neurons=512,
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=4096,
        out_neurons=128,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        self.blocks.extend(
            [torch.nn.Linear(inp_neurons, lin_neurons), activation()]
        )
        for block_index in range(1, lin_blocks):
            self.blocks.extend(
                [torch.nn.Linear(lin_neurons, lin_neurons), activation()]
            )

        self.blocks.extend([torch.nn.Linear(lin_neurons, out_neurons)])

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x


class ContrastiveLoss(torch.nn.Module):
    """
    An implementation of contrastive loss with a trainable projection. Because
    of dot products, predictions are symmetric.

    Arguments
    ---------
    temperature: float
        The temperature for scaling logits before softmax

    Return
    ---------
    predictions : torch.Tensor
        The probabilities of samples of shape [N, N], where N is the batch size.
        The diagonal elements should be ignored.

    Example
    -------
    >>> projection = Projection(inp_neurons=2, lin_neurons=2048, out_neurons=128)
    >>> prob = ContrastiveLoss(projection, temperature=0.1)
    >>> outputs = torch.tensor([ [1., 2.], [-1., -2.], [2., 1.], [-2., -1.] ])
    >>> prediction = prob(outputs)
    """

    def __init__(self, projection, device="cpu", temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.projection = projection.to(device)
        self.temperature = temperature

    def forward(self, outputs):
        """
        Compute contrastive loss for outputs

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, D]

        Return
        ---------
        predictions : torch.Tensor
            Log probabilities of samples of shape [N, N], where N is the batch size.
        """
        outputs = self.projection(outputs)
        outputs = F.normalize(outputs)
        N = outputs.shape[0]
        logits = torch.matmul(outputs, outputs.transpose(0, 1))
        neg_identity = -1e6 * torch.eye(N).to(outputs.device)
        logits = logits.reshape(N, N) + neg_identity
        return F.softmax(logits / self.temperature, dim=1)


class BinaryContrastiveLoss(ContrastiveLoss):
    """
    An implementation of binary contrastive loss with a trainable projection.
    It concatenates two embeddings, and predict whether they belong to the same
    class or not. Therefore, predictions may not be symmetric.

    Arguments
    ---------
    projection: torch.nn.Module
        Predicts logits for concatenated embeddings

    temperature: float
        The temperature for scaling logits before softmax

    Return
    ---------
    predictions : torch.Tensor
        The probabilities of samples of shape [N, N], where N is the batch size.
        The diagonal elements should be ignored.

    Example
    -------
    >>> projection = Projection(inp_neurons=2+2, out_neurons=1)
    >>> prob = BinaryContrastiveLoss(projection, temperature=0.1)
    >>> outputs = torch.tensor([ [1., 2.], [2., 1.], [-1., -2.], [-2., -1.] ])
    >>> prediction = prob(outputs)
    """

    def forward(self, outputs):
        """
        Compute Discriminative ContrastiveLoss between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, D]

        Return
        ---------
        predictions : torch.Tensor

        """
        N = outputs.shape[0]
        outputs = outputs.unsqueeze(1).repeat(1, N, 1)
        outputs = torch.cat([outputs, outputs.transpose(0, 1)], dim=2)
        neg_identity = -1e6 * torch.eye(N).to(outputs.device)
        logits = self.projection(outputs).reshape(N, N) + neg_identity
        return logits / self.temperature


class ContrastiveLearningWrapper(torch.nn.Module):
    """
    Contrastive learning

    Arguments
    ---------
    cont_loss : torch.nn.Module
        User-defined module that compute the probabilities of outputs

    criterion: torch.nn.{BCELoss,BCEWithLogitsLoss}
        Compute the loss between predictions and targets, where reduction must
        be 'none' for distinguishing positive/negative losses

    Returns
    ---------
    loss : torch.Tensor
        Contrastive learning loss

    predictions : torch.Tensor
        The probabilities of outputs

    Example
    -------
    >>> outputs = torch.tensor([ [1., 2.], [-1., -2.], [2., 1.], [-2., -1.] ])
    >>> outputs = outputs.unsqueeze(1)
    >>> targets = torch.tensor([   1,        2,         1,          2,      ])
    >>> targets = targets.unsqueeze(1)
    >>> projection = Projection(inp_neurons=2, out_neurons=128)
    >>> criterion = torch.nn.BCELoss(reduction="none")
    >>> cont_loss = ContrastiveLoss(projection, temperature=0.1)
    >>> cont = ContrastiveLearningWrapper(cont_loss, criterion)
    >>> optimizer = torch.optim.SGD(projection.parameters(), lr=0.01)
    >>> for i in range(25):
    ...     loss, predictions = cont(outputs, targets)
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()
    >>> torch.argmax(predictions, dim=1)
    tensor([2, 3, 0, 1])
    >>> projection = Projection(inp_neurons=2+2, out_neurons=1)
    >>> criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    >>> cont_loss = BinaryContrastiveLoss(projection, temperature=0.1)
    >>> cont = ContrastiveLearningWrapper(cont_loss, criterion, nl_weight=0.5)
    >>> optimizer = torch.optim.SGD(projection.parameters(), lr=0.01)
    >>> for i in range(25):
    ...     loss, predictions = cont(outputs, targets)
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()
    >>> torch.argmax(predictions, dim=1)
    tensor([2, 3, 0, 1])
    """

    def __init__(self, cont_loss, criterion, nl_weight=0.0):
        super(ContrastiveLearningWrapper, self).__init__()
        self.cont_loss = cont_loss
        self.criterion = criterion
        self.nl_weight = nl_weight

    def forward(self, outputs, targets):
        """
            Arguments
            ---------
            outputs : torch.Tensor
                Network output tensor, of shape
                [batch, 1, outdim, ...].
            targets : torch.Tensor
                Target tensor, of shape [batch, 1].

            Returns
            -------
            loss: torch.Tensor
                contrastive learning loss
        """
        N = outputs.shape[0]
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        predictions = self.cont_loss(outputs)
        positives = torch.zeros([N, N]).to(outputs.device)
        for y in targets:
            index = (targets == y).nonzero(as_tuple=False)
            for i in index:
                for j in index:
                    if i != j:
                        positives[i, j] = 1
        # TODO: dealing with more than 1 positive per anchor?
        loss_matrix = self.criterion(predictions, positives)
        positive_loss = loss_matrix * positives
        positive_loss = positive_loss.sum() / positives.sum()
        loss = positive_loss
        if self.nl_weight > 0:
            negatives = (1 - positives) * (1 - torch.eye(N).to(outputs.device))
            negative_loss = loss_matrix * negatives
            negative_loss = negative_loss.sum() / negatives.sum()
            loss += self.nl_weight * negative_loss
        return loss, predictions
