"""Map a flat dimension with points of rank N, according to the Hilbert curve."""

import math

import numpy
import torch

import densecurves.hilbert
import mlable.shapes

# 1D PERMUTATION ###############################################################

def permutation(order: int, rank: int, group: int=0, flatten: bool=False) -> list:
    # 1D dimension of the curve: 2 ** (order * rank)
    __dim = 1 << (order + group) * rank
    # target shape: (2 ** order, 2 ** order, ...) rank times
    __shape = rank * [1 << (order + group)]
    # the whole list of vertexes
    __curve = [densecurves.hilbert.point(__i, order=order, rank=rank, group=group) for __i in range(__dim)]
    # match the format of numpy: one row per dimension (!= one row per point)
    __indices = list(zip(*__curve))
    # permutation: __flat[i] contains the destination index of i
    __flat = numpy.ravel_multi_index(__indices, dims=__shape, mode='wrap', order='C')
    # mapping between destination and origin indices
    __map = {(__o if flatten else __d): (__d if flatten else __o) for __o, __d in enumerate(__flat.tolist())}
    # match the format of gather: __perm[i] contains the origin of index i
    return [__map[__d] for __d in sorted(__map.keys())]

# 1D => ND #####################################################################

def fold(data: torch.Tensor, order: int, rank: int, axis: int, group: int=0) -> torch.Tensor:
    # only integer dimension (0 for None)
    __shape = mlable.shapes.normalize(data.shape)
    # avoid negative indices => axis + 1 != 0
    __axis = axis % len(__shape)
    # insert the new axes
    __shape = __shape[:__axis] + rank * [1 << (order + group)] + __shape[__axis + 1:]
    # 1D reordering of the indexes according to the Hilbert curve
    __perm = permutation(order=order, rank=rank, group=group, flatten=False)
    # actually swap the elements along the target axis
    __data = torch.index_select(data, dim=__axis, index=torch.tensor(__perm, device=data.device, dtype=torch.long))
    # split the sequence axis
    return __data.reshape(__shape)

# ND => 1D #####################################################################

def unfold(data: torch.Tensor, order: int, rank: int, axes: iter, group: int=0) -> torch.Tensor:
    # only integer dimension (0 for None)
    __shape = mlable.shapes.normalize(data.shape)
    # only positive indexes
    __axes = sorted([__a % len(__shape) for __a in axes])
    # merge all the axes on the first one
    __shape = __shape[:min(__axes)] + [math.prod(__shape[min(__axes):max(__axes) + 1])] + __shape[max(__axes) + 1:]
    # 1D permutation of the indexes along the merged axis
    __perm = permutation(order=order, rank=rank, group=group, flatten=True)
    # merge all the axes of the hypercube
    __data = data.reshape(__shape)
    # actually swap the elements along the target axis
    return torch.index_select(__data, dim=min(axes), index=torch.tensor(__perm, device=data.device, dtype=torch.long))
