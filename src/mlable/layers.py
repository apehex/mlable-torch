import math

import torch
import torch.nn

import mlable.shapes

# RESHAPING ###################################################################

def _normalize_shape(shape: list) -> list:
    return [-1 if __d is None else __d for __d in shape]

def _normalize_dim(dim: int) -> int:
    return -1 if (dim is None or dim < 0) else dim

def _multiply_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1) else dim_l * dim_r

def _divide_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1) else dim_l // dim_r

class Divide(torch.nn.Module):
    def __init__(
        self,
        input_axis: int, # relative to the NEW shape / rank
        output_axis: int, # same
        factor: int,
        insert: bool=False,
        **kwargs
    ) -> None:
        super(Divide, self).__init__(**kwargs)
        self._input_axis = input_axis
        self._output_axis = output_axis
        self._factor = factor
        self._insert = insert

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # infer the dimension of the symbolic axis
        __shape = _normalize_shape(list(inputs.shape))
        # rank, according to the new shape
        __rank = len(__shape) + int(self._insert)
        # axes, taken from the new shape
        __axis0 = self._input_axis % __rank
        __axis1 = self._output_axis % __rank
        # option to group data on a new axis
        if self._insert: __shape.insert(__axis1, 1)
        # move data from axis 0 to axis 1
        __shape[__axis0] = _divide_dim(__shape[__axis0], self._factor)
        __shape[__axis1] = _multiply_dim(__shape[__axis1], self._factor)
        return inputs.view(*__shape) #.squeeze(1)

class Merge(torch.nn.Module):
    def __init__(
        self,
        left_axis: int=-2,
        right_axis: int=-1,
        left: bool=True,
        **kwargs
    ) -> None:
        super(Merge, self).__init__(**kwargs)
        self._left_axis = left_axis
        self._right_axis = right_axis
        self._left = left

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # infer the dimension of the symbolic axis
        __shape = _normalize_shape(list(inputs.shape))
        __rank = len(__shape)
        # target axes
        __axis_l = self._left_axis % __rank
        __axis_r = self._right_axis % __rank
        # new axis
        __dim = _multiply_dim(__shape[__axis_l], __shape[__axis_r])
        __axis_k = __axis_l if self._left else __axis_r # kept axis
        __axis_d = __axis_r if self._left else __axis_l # deleted axis
        # new shape
        __shape[__axis_k] = __dim
        __shape.pop(__axis_d)
        # actually merge the two axes
        return inputs.view(*__shape)

# LINEAR ######################################################################

class PositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        time_dim: int,
        embed_dim: int,
        input_axis: int=1, # axis of the sequence
        output_axis: int=-1, # axis of the embedding
        **kwargs
    ):
        super(PositionalEmbedding, self).__init__(**kwargs)
        # weights
        self._input_axis = input_axis
        self._output_axis = output_axis
        self._kernel = torch.nn.Parameter(torch.randn((time_dim, embed_dim)), requires_grad=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # shape
        __input_shape = list(inputs.shape)
        __axes = [self._input_axis % len(__input_shape), self._output_axis % len(__input_shape)]
        __output_shape = [(__d if __i in __axes else 1) for __i, __d in enumerate(list(__input_shape))]
        return inputs + self._kernel.view(*__output_shape) # each index in the sequence axis has a dedicated bias (different from dense bias)

class CompositeEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        group_dim: int=-1,
        merge_axes: bool=True,
        **kwargs
    ) -> None:
        super(CompositeEmbedding, self).__init__(
            num_embeddings=input_dim,
            embedding_dim=output_dim,
            **kwargs)
        # save for import / export
        self._config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'group_dim': -1 if group_dim is None else group_dim,
            'merge_axes': merge_axes,
            **kwargs}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        __group = self._config.get('group_dim', -1)
        __merge = self._config.get('merge_axes', True)
        # split the last axis in blocks of fixed dimension
        __shape = mlable.shapes.divide(
            shape=tuple(inputs.shape),
            axis=-1,
            factor=max(1, __group),
            insert=True,
            right=True)
        # leave the shape unchanged if the group dimension is negative (..., S*G) => (..., S, G)
        __outputs = inputs.reshape(tuple(inputs.shape) if (__group <= 1) else __shape)
        # embed the input IDs (..., S, G) -> (..., S, G, E)
        __outputs = super(CompositeEmbedding, self).forward(__outputs)
        # merge the last 2 axes (..., S, G, E) -> (..., S, G*E)
        __shape = mlable.shapes.merge(shape=tuple(__outputs.shape), axis=-1, right=False)
        # group only if requested
        return __outputs.reshape(__shape if __merge else tuple(__outputs.shape))
