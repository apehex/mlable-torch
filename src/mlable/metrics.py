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
    # expand the shape of the mask with singleton axes
    __shape = mlable.shapes.filter(tuple(__outputs.shape), axes=list(range(mask_arr.ndim)))
    # match the rank and dtype for the multiplications
    __weights = mask_arr.reshape(__shape).float()
    # average over the elements in the mask only: N_tot / N_mask
    __factor = float(math.prod(tuple(__weights.shape))) / max(1.0, float(__weights.sum()))
    # discard the values outside of the mask
    return __factor * (__outputs * __weights).mean()