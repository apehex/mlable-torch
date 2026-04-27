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
            'input_axis': int(input_axis),
            'output_axis': int(output_axis),}
        # build at runtime
        self._kernel = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None,
    ) -> None:
        # lazy build at runtime
        if (not self._built) or (self._kernel is None):
            # parse the inputs
            __rank = len(shape)
            # normalize the indexes
            __axis_i = self._config['input_axis'] % __rank
            __axis_o = self._config['output_axis'] % __rank
            # handle the case where feature axis comes before the sequence axis
            __dim_i = shape[min(__axis_i, __axis_o)]
            __dim_o = shape[max(__axis_i, __axis_o)]
            # built the kernel
            self._kernel = torch.nn.Parameter(
                torch.randn((__dim_i, __dim_o), dtype=dtype, device=device),
                requires_grad=True)
            # register
            self._built = True

    def forward(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        # parse the inputs
        __shape = tuple(inputs.shape)
        # create the kernel, if necessary
        self.build(shape=__shape, device=inputs.device, dtype=inputs.dtype)
        # where to apply the positional embedding
        __axes = [self._config['input_axis'], self._config['output_axis']]
        # extend the shape of the kernel to match the rank of the inputs
        __shape = mlable.shapes.filter(__shape, axes=__axes)
        # each index in the sequence axis has a dedicated bias (different from dense bias)
        return inputs + self._kernel.view(*__shape)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self._kernel)

    def output_shape(self, shape: tuple) -> tuple:
        return tuple(shape)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})

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
            'input_dim': int(input_dim),
            'output_dim': int(output_dim),
            'group_dim': -1 if group_dim is None else int(group_dim),
            'merge_axes': bool(merge_axes),}
        # register
        self._built = True

    def build(self, shape: tuple=(), device: object=None, dtype: object=None) -> None:
        self.to(dtype=dtype, device=device)
        self._built = True

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
            insert=bool(__group > 1),
            right=bool(__group > 1))
        # leave the shape unchanged if the group dimension is negative (..., S*G) => (..., S, G)
        __outputs = inputs.reshape(__shape)
        # embed the input IDs (..., S, G) -> (..., S, G, E)
        __outputs = super(CompositeEmbedding, self).forward(__outputs)
        # merge the last 2 axes (..., S, G, E) -> (..., S, G*E)
        __shape = mlable.shapes.merge(shape=tuple(__outputs.shape), axis=-1, right=False)
        # combine only if requested
        return __outputs.reshape(__shape if __merge else tuple(__outputs.shape))

    def output_shape(self, shape: tuple) -> tuple:
        __embed = self._config.get('output_dim', 1)
        __group = self._config.get('group_dim', -1)
        __merge = self._config.get('merge_axes', True)
        # split the last axis in blocks of fixed dimension, if requested (..., S*G) => (..., S, G)
        __shape = mlable.shapes.divide(
            shape=tuple(shape),
            axis=-1,
            factor=max(1, __group),
            insert=bool(__group > 1),
            right=bool(__group > 1))
        # embed the inputs (..., S, G) => (..., S, G, E)
        __shape = list(__shape) + [__embed] + (not __merge) * [1]
        # merge the last 2 axes, if requested (..., S, G, E) -> (..., S, G*E)
        return tuple(mlable.shapes.merge(shape=__shape, axis=-1, right=False))

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})
