import math

import torch
import torch.nn
import torch.nn.functional

import mlable.shapes
import mlable.shaping.axes

# SwiGLU #######################################################################

class GatedLinearUnit(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        affine_opt: bool=True,
        **kwargs: dict
    ) -> None:
        super(GatedLinearUnit, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'hidden_dim': int(hidden_dim),
            'output_dim': int(output_dim),
            'affine_opt': bool(affine_opt),}
        # build at runtime
        self._extend = None
        self._project = None
        self._built = False

    def build(
        self,
        shape: tuple,
        dtype: object=None,
        device: object=None
    ) -> None:
        if not self._built:
            # (..., E) => (..., 2*H)
            self._extend = torch.nn.Linear(
                in_features=int(shape[-1]),
                out_features=2 * self._config['hidden_dim'],
                bias=self._config['affine_opt']).to(dtype=dtype, device=device)
            # (..., H) => (..., O)
            self._project = torch.nn.Linear(
                in_features=self._config['hidden_dim'],
                out_features=self._config['output_dim'],
                bias=self._config['affine_opt']).to(dtype=dtype, device=device)
            # register
            self._built = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # lazy build
        self.build(shape=tuple(inputs.shape), dtype=inputs.dtype, device=inputs.device)
        # generate both the gate and value activations at once
        __gate, __value = self._extend(inputs).chunk(chunks=2, dim=-1)
        # filter the values and project on the output dimension
        return self._project(torch.nn.functional.silu(__gate) * __value)

    def output_shape(self, shape: tuple) -> tuple:
        return tuple(shape)[:-1] + (self._config['output_dim'],)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})

# SELF-ATTENTION ###############################################################

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        head_num: int,
        attention_idx: int=-2,
        affine_opt: bool=True,
        dropout_rate: float=0.0,
        **kwargs
    ) -> None:
        super(SelfAttention, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'head_num': int(head_num),
            'attention_idx': int(attention_idx) if isinstance(attention_idx, int) else -2,
            'affine_opt': bool(affine_opt) if isinstance(affine_opt, bool) else True,
            'dropout_rate': float(dropout_rate) if isinstance(dropout_rate, float) else 0.0,
            **kwargs}
        # build at runtime
        self._layer = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None
    ) -> None:
        if (not self._built) or (self._layer is None):
            # init the layer / weights
            self._layer = torch.nn.MultiheadAttention(
                embed_dim=int(shape[-1]),
                num_heads=self._config['head_num'],
                bias=self._config['affine_opt'],
                dropout=self._config['dropout_rate'],
                kdim=None,
                vdim=None,
                batch_first=True,
                add_bias_kv=False,
                add_zero_attn=False).to(dtype=dtype, device=device)
            # register
            self._built = True

    def forward(
        self,
        inputs: torch.Tensor,
        paddings: torch.Tensor=None,
        **kwargs: dict,
    ) -> torch.Tensor:
        # at least rank 3 to have batch, attention and feature axis
        assert (len(inputs.shape) > 2), 'Inputs must have distinct batch, sequence and feature axes.'
        # check the mask when provided
        if paddings is not None:
            # the inputs and mask shapes must match
            assert (tuple(paddings.shape) == tuple(inputs.shape)[:-1]), f'Inputs are {inputs.shape} while the mask is {paddings.shape}.'
        # lazy build
        self.build(shape=tuple(inputs.shape), dtype=inputs.dtype, device=inputs.device)
        # store the intermediate shape to split the batch axes back
        __shape = self.swapped_shape(inputs.shape)
        # move the attention axis and merge the batch axes into (B, A, F)
        __outputs = self.preprocess(inputs)
        # perform the same operation on the mask 
        __mask = (
            None if not hasattr(paddings, 'shape')
            else self.preprocess(paddings.unsqueeze(-1)).squeeze(-1).to(dtype=torch.bool))
        # finally apply the attention
        __outputs, _ = self._layer(
            query=__outputs,
            key=__outputs,
            value=__outputs,
            key_padding_mask=__mask,
            need_weights=False,
            **kwargs)
        # restore the axes
        return self.postprocess(__outputs, shape=__shape)

    def preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        # move the attention axis so that the tensor is (..., A, F)
        __outputs = mlable.shaping.axes.swap(
            data=inputs,
            left_axis=self._config['attention_idx'],
            right_axis=-2).contiguous()
        # merge all the batch axes, so that the tensor is rank 3 (B, A, F)
        return __outputs.reshape(self.merged_shape(__outputs.shape))

    def postprocess(self, inputs: torch.Tensor, shape: tuple) -> torch.Tensor:
        # restore the batch axes
        __outputs = inputs.reshape(shape)
        # move the attention axis back
        return mlable.shaping.axes.swap(
            data=__outputs,
            left_axis=self._config['attention_idx'],
            right_axis=-2).contiguous()

    def swapped_shape(self, shape: tuple) -> tuple:
        return mlable.shapes.swap(shape, left=self._config['attention_idx'], right=-2)

    def merged_shape(self, shape: tuple) -> tuple:
        return (math.prod(tuple(shape[:-2])),) + tuple(shape[-2:])

    def output_shape(self, shape: tuple) -> tuple:
        return tuple(shape)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})
