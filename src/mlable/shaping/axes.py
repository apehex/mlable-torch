import torch

import mlable.shapes

# DIVIDE ######################################################################

def divide(data: torch.Tensor, axis: int, factor: int, insert: bool=False, right: bool=True) -> torch.Tensor:
    # move data from the source axis to its neighbor
    __shape = mlable.shapes.divide(shape=list(data.shape), axis=axis, factor=factor, insert=insert, right=right)
    # actually reshape
    return data.reshape(*__shape)

# MERGE #######################################################################

def merge(data: torch.Tensor, axis: int, right: bool=True) -> torch.Tensor:
    # new shape
    __shape = mlable.shapes.merge(shape=list(data.shape), axis=axis, right=right)
    # actually merge the two axes
    return data.reshape(*__shape)

# SWAP #########################################################################

def swap(data: torch.Tensor, left_axis: int, right_axis: int) -> torch.Tensor:
    # mapping from the new axis indices to the old indices
    __perm = mlable.shapes.swap(shape=range(len(data.shape)), left=left_axis, right=right_axis)
    # transpose the data instead of just reshaping
    return data.permute(*__perm)

# MOVE #########################################################################

def move(data: torch.Tensor, from_axis: int, to_axis: int) -> torch.Tensor:
    # mapping from the new axis indices to the old indices
    __perm = mlable.shapes.move(shape=range(len(data.shape)), before=from_axis, after=to_axis)
    # transpose the data instead of just reshaping
    return data.permute(*__perm)
