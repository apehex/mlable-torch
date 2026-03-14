import torch
import torch.nn

import mlable.shapes

# LEARNED POSITIONS ############################################################

class PositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        time_dim: int,
        embed_dim: int,
        input_axis: int=1, # axis of the sequence
        output_axis: int=-1, # axis of the embedding
        **kwargs
    ) -> None:
        super(PositionalEmbedding, self).__init__(**kwargs)
        # weights
        self._input_axis = input_axis
        self._output_axis = output_axis
        self._kernel = torch.nn.Parameter(torch.randn((time_dim, embed_dim)), requires_grad=True)

    def forward(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        # shape
        __input_shape = list(inputs.shape)
        __axes = [self._input_axis % len(__input_shape), self._output_axis % len(__input_shape)]
        __output_shape = [(__d if __i in __axes else 1) for __i, __d in enumerate(list(__input_shape))]
        return inputs + self._kernel.view(*__output_shape) # each index in the sequence axis has a dedicated bias (different from dense bias)

# TOKUN ########################################################################

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

    def forward(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
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
