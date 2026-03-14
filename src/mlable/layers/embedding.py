import torch
import torch.nn

import mlable.shapes

# LEARNED POSITIONS ############################################################

class PositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        input_axis: int=1, # axis of the sequence
        output_axis: int=-1, # axis of the embedding
        **kwargs
    ) -> None:
        super(PositionalEmbedding, self).__init__(**kwargs)
        # save for import, export, duplication etc
        self._config = {
            'input_axis': input_axis,
            'output_axis': output_axis,}
        # build at runtime
        self._kernel = None
        self._built = False

    def _build(
        self,
        inputs: torch.Tensor
    ) -> None:
        # lazy build at runtime
        if (not self._built) or (self._kernel is None):
            # parse the inputs
            __shape = tuple(inputs.shape)
            __rank = len(__shape)
            # normalize the indexes
            __axis_i = self._config['input_axis'] % __rank
            __axis_o = self._config['output_axis'] % __rank
            # handle the case where feature axis comes before the sequence axis
            __dim_i = __shape[min(__axis_i, __axis_o)]
            __dim_o = __shape[max(__axis_i, __axis_o)]
            # built the kernel
            self._kernel = torch.nn.Parameter(
                torch.randn((__dim_i, __dim_o)),
                device=inputs.device,
                dtype=inputs.dtype,
                requires_grad=True)
            # register
            self._built = True

    def forward(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        # create the kernel, if necessary
        self._build(inputs)
        # parse the inputs
        __shape = list(inputs.shape)
        # where to apply the positional embedding
        __axes = [self._config['input_axis'], self._config['output_axis']]
        # extend the shape of the kernel to match the rank of the inputs
        __shape = mlable.shapes.filter(__shape, axes=__axes)
        # each index in the sequence axis has a dedicated bias (different from dense bias)
        return inputs + self._kernel.view(*__shape)

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
