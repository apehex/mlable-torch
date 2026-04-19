import math

import torch
import torch.nn.functional

import mlable.shapes

# MSE ##########################################################################

def mse_loss(
    predict_arr: torch.Tensor,
    target_arr: torch.Tensor,
    mask_arr: torch.Tensor,
) -> torch.Tensor:
    """KL divergence over (B, T, V) raw logits with (B, T) mask."""
    # expand the shape of the mask with singleton axes
    __shape = mlable.shapes.filter(tuple(target_arr.shape), axes=list(range(mask_arr.ndim)))
    # match the rank and dtype for the multiplications
    __weights = mask_arr.reshape(__shape).float()
    # average over the elements in the mask only: N_tot / N_mask
    __factor = float(math.prod(tuple(__weights.shape))) / max(1.0, float(__weights.sum()))
    # zero the elements outside of the mask
    __preds = predict_arr.float() * __weights
    __targs = target_arr.float() * __weights
    # reduce to a single value
    return __factor * torch.nn.functional.mse_loss(
        input=__preds,
        target=__targs,
        weight=None,
        reduction='mean')

# KL-DIV #######################################################################

def kl_div(
    predict_arr: torch.Tensor,
    target_arr: torch.Tensor,
    mask_arr: torch.Tensor,
) -> torch.Tensor:
    """KL divergence over (B, T, V) raw logits with (B, T) mask."""
    __shape = tuple(target_arr.shape)
    # match the rank and dtype for the multiplications
    __weights = mask_arr.reshape(mlable.shapes.filter(__shape, axes=list(range(mask_arr.ndim))))
    __weights = __weights.float()
    # average over the elements in the mask only: N_tot / N_mask
    __factor = float(math.prod(tuple(__weights.shape))) / max(1.0, float(__weights.sum()))
    # zero the elements outside of the mask
    __preds = predict_arr.float() * __weights
    __targs = target_arr.float() * __weights
    # merge the batch axes
    __preds = __preds.reshape(math.prod(__shape[:-1]), __shape[-1])
    __targs = __targs.reshape(math.prod(__shape[:-1]), __shape[-1])
    # reduce to a single value
    return __factor * torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(__preds, dim=-1),
        torch.nn.functional.log_softmax(__targs, dim=-1),
        reduction='batchmean',
        log_target=True)
