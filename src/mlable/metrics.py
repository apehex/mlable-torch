import math

import torch
import torch.nn.functional

import mlable.shapes

# TOP-K ########################################################################

def topk_rate(
    predict_arr: torch.Tensor,
    target_arr: torch.Tensor,
    mask_arr: torch.Tensor,
    k_num: int=10,
) -> torch.Tensor:
    """Fraction of (B, T) positions where teacher top-k and student top-k token sequences match exactly."""
    __preds = predict_arr.topk(k_num, dim=-1, sorted=True).indices
    __targs = target_arr.topk(k_num, dim=-1, sorted=True).indices
    # count the positions where all the top-k indices match
    __outputs = (__preds == __targs).all(dim=-1).float()
    # match the rank and dtype for the multiplications
    __mask = mask_arr.reshape(mlable.shapes.filter(
        shape=tuple(__outputs.shape),
        axes=list(range(mask_arr.ndim)))).float()
    # calculate the average over the masked positions only
    return (__outputs * __mask).sum() / __mask.sum().clamp_min(1.0)
