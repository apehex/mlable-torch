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
    """MSE over (B, T, H) features with (B, T) mask."""
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

# COSINE #######################################################################

def cos_sim(
    predict_arr: torch.Tensor,
    target_arr: torch.Tensor,
    mask_arr: torch.Tensor,
) -> torch.Tensor:
    """Masked mean cosine similarity over (B, T, H) tensors."""
    # match the rank and dtype for the multiplications
    __mask = mask_arr.reshape(mlable.shapes.filter(
        shape=tuple(target_arr.shape),
        axes=list(range(mask_arr.ndim)))).float()
    # compute the point-wise cosine similarity
    __outputs = torch.nn.functional.cosine_similarity(
        x1=predict_arr.float(),
        x2=target_arr.float(),
        dim=-1)
    # calculate the batch mean, over the masked positions
    return (__outputs * __mask).sum() / __mask.sum().clamp(min=1.0)

# KL-DIV #######################################################################

def kl_div(
    predict_arr: torch.Tensor,
    target_arr: torch.Tensor,
    mask_arr: torch.Tensor,
) -> torch.Tensor:
    """KL divergence over (B, T, V) raw logits with (B, T) mask."""
    # match the rank and dtype for the multiplications
    __mask = mask_arr.reshape(mlable.shapes.filter(
        shape=tuple(target_arr.shape),
        axes=list(range(mask_arr.ndim)))).float()
    # compute the point-wise KL-divergence
    __outputs = torch.nn.functional.kl_div(
        input=torch.nn.functional.log_softmax(predict_arr.float(), dim=-1),
        target=torch.nn.functional.log_softmax(target_arr.float(), dim=-1),
        reduction='none',
        log_target=True)
    # calculate the batch mean, over the masked positions only
    return (__outputs * __mask).sum() / __mask.sum().clamp_min(1.0)
