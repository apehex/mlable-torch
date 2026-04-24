import math

import torch
import torch.nn.functional

import mlable.shapes

# TOP-K ########################################################################

def topk_rate(
    predict_arr: torch.Tensor,
    target_arr: torch.Tensor,
    mask_arr: torch.Tensor=None,
    reduce_opt: bool=True,
    k_num: int=10,
) -> torch.Tensor:
    """Fraction of (B, T) positions where teacher top-k and student top-k token sequences match exactly."""
    __k = min(k_num, predict_arr.shape[-1])
    __preds = predict_arr.topk(__k, dim=-1).indices
    __targs = target_arr.topk(__k, dim=-1).indices
    # (B, T, K) target indices that appear in the predictions
    __outputs = (__targs.unsqueeze(-1) == __preds.unsqueeze(-2)).any(dim=-1)
    # (B, T) fractions of the top-k that overlap
    __outputs = __outputs.float().mean(dim=-1)
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
