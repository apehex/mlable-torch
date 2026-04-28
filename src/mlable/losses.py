import math

import torch
import torch.nn.functional

import mlable.shapes

# MSE ##########################################################################

def mse_loss(
    predict_arr: torch.Tensor,
    target_arr: torch.Tensor,
    mask_arr: torch.Tensor=None,
    reduce_opt: bool=True,
    relative_opt: bool=False,
    epsilon_rate: float=1e-8,
) -> torch.Tensor:
    """MSE over (B, T, H) features with (B, T) mask."""
    # compute the element-wise MSE
    __outputs = torch.nn.functional.mse_loss(
        input=predict_arr.float(),
        target=target_arr.float(),
        weight=None,
        reduction='none').mean(dim=-1)
    # scale the loss according to the target
    if relative_opt:
        __scale = (target_arr.float() ** 2).mean(dim=-1).clamp(min=epsilon_rate)
        __outputs = __outputs / __scale
    # include all the positions by default
    __mask = (
        mask_arr if hasattr(mask_arr, 'ndim')
        else torch.ones(__outputs.shape, dtype=__outputs.dtype, device=__outputs.device))
    # match the rank and dtype for the multiplications
    __mask = __mask.reshape(mlable.shapes.filter(
        shape=tuple(__outputs.shape),
        axes=list(range(__mask.ndim)))).float()
    # filter the results for the positions outside of the mask
    __outputs = __outputs * __mask
    # average over the masked positions only, if requested
    return (__outputs.sum() / __mask.sum().clamp_min(1.0)) if reduce_opt else __outputs

# COSINE #######################################################################

def cos_sim(
    predict_arr: torch.Tensor,
    target_arr: torch.Tensor,
    mask_arr: torch.Tensor=None,
    reduce_opt: bool=True,
) -> torch.Tensor:
    """Masked mean cosine similarity over (B, T, H) tensors."""
    # compute the point-wise cosine similarity
    __outputs = torch.nn.functional.cosine_similarity(
        x1=predict_arr.float(),
        x2=target_arr.float(),
        dim=-1)
    # include all the positions by default
    __mask = (
        mask_arr if hasattr(mask_arr, 'ndim')
        else torch.ones(__outputs.shape, dtype=__outputs.dtype, device=__outputs.device))
    # match the rank and dtype for the multiplications
    __mask = __mask.reshape(mlable.shapes.filter(
        shape=tuple(__outputs.shape),
        axes=list(range(__mask.ndim)))).float()
    # filter the results for the positions outside of the mask
    __outputs = __outputs * __mask
    # average over the masked positions only, if requested
    return (__outputs.sum() / __mask.sum().clamp_min(1.0)) if reduce_opt else __outputs

# KL-DIV #######################################################################

def kl_div(
    predict_arr: torch.Tensor,
    target_arr: torch.Tensor,
    mask_arr: torch.Tensor=None,
    reduce_opt: bool=True,
) -> torch.Tensor:
    """KL divergence over (B, T, V) raw logits with (B, T) mask."""
    # compute the point-wise KL-divergence
    __outputs = torch.nn.functional.kl_div(
        input=torch.nn.functional.log_softmax(predict_arr.float(), dim=-1),
        target=torch.nn.functional.log_softmax(target_arr.float(), dim=-1),
        reduction='none',
        log_target=True).sum(dim=-1)
    # include all the positions by default
    __mask = (
        mask_arr if hasattr(mask_arr, 'ndim')
        else torch.ones(__outputs.shape, dtype=__outputs.dtype, device=__outputs.device))
    # match the rank and dtype for the multiplications
    __mask = __mask.reshape(mlable.shapes.filter(
        shape=tuple(__outputs.shape),
        axes=list(range(__mask.ndim)))).float()
    # filter the results for the positions outside of the mask
    __outputs = __outputs * __mask
    # average over the masked positions only, if requested
    return (__outputs.sum() / __mask.sum().clamp_min(1.0)) if reduce_opt else __outputs
