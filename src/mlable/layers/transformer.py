import math

import torch
import torch.nn
import torch.nn.functional

# SELF ATTENTION ###############################################################

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        head_num: int,
        head_dim: int,
        causal_opt: bool=True,
        **kwargs: dict,
    ) -> None:
        super(SelfAttention, self).__init__(**kwargs)
        # save for import, export, duplication etc
        self._config = {
            'head_num': int(head_num),
            'head_dim': int(head_dim),
            'causal_opt': bool(causal_opt),}
        # build at runtime
        self._wq = None
        self._wk = None
        self._wv = None
        self._wo = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None,
    ) -> None:
        if not self._built:
            __embed = self._config['head_num'] * self._config['head_dim']
            __input_dim = int(shape[-1])
            self._wq = torch.nn.Parameter(torch.empty(__input_dim, __embed, device=device, dtype=dtype))
            self._wk = torch.nn.Parameter(torch.empty(__input_dim, __embed, device=device, dtype=dtype))
            self._wv = torch.nn.Parameter(torch.empty(__input_dim, __embed, device=device, dtype=dtype))
            self._wo = torch.nn.Parameter(torch.empty(__embed, __input_dim, device=device, dtype=dtype))
            self.reset_parameters()
            self._built = True

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor=None) -> torch.Tensor:
        __shape = tuple(inputs.shape)
        self.build(shape=__shape, device=inputs.device, dtype=inputs.dtype)
        __b, __t = __shape[0], __shape[1]
        __h = self._config['head_num']
        __d = self._config['head_dim']
        __scale = math.sqrt(__d)
        # projections: (B, T, H*D)
        __q = inputs @ self._wq
        __k = inputs @ self._wk
        __v = inputs @ self._wv
        # multi-head reshape: (B, H, T, D)
        __q = __q.view(__b, __t, __h, __d).transpose(1, 2)
        __k = __k.view(__b, __t, __h, __d).transpose(1, 2)
        __v = __v.view(__b, __t, __h, __d).transpose(1, 2)
        # scaled dot-product attention logits: (B, H, T, T)
        __logits = torch.matmul(__q, __k.transpose(-2, -1)) / __scale
        # causal mask: block attention to future positions
        if self._config['causal_opt']:
            __causal_mask = torch.ones(__t, __t, device=inputs.device, dtype=torch.bool).triu(1)
            __logits = __logits.masked_fill(__causal_mask, float('-inf'))
        # padding mask: block attention to padding key positions
        if paddings is not None:
            # (B, T) -> (B, 1, 1, T): any query cannot attend to a padding key
            __pad_mask = paddings.bool().unsqueeze(1).unsqueeze(2)
            __logits = __logits.masked_fill(__pad_mask, float('-inf'))
        # softmax over key axis and weighted sum
        __weights = torch.softmax(__logits, dim=-1)
        __context = torch.matmul(__weights, __v)
        # recombine heads: (B, T, H*D)
        __context = __context.transpose(1, 2).contiguous().view(__b, __t, __h * __d)
        # output projection: (B, T, E_in)
        return __context @ self._wo

    def reset_parameters(self) -> None:
        if self._wq is not None:
            torch.nn.init.xavier_uniform_(self._wq)
            torch.nn.init.xavier_uniform_(self._wk)
            torch.nn.init.xavier_uniform_(self._wv)
            torch.nn.init.xavier_uniform_(self._wo)

    def output_shape(self, shape: tuple) -> tuple:
        return tuple(shape)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})

# TRANSFORMER BLOCK ############################################################

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        head_num: int,
        head_dim: int,
        mlp_dim: int,
        causal_opt: bool=True,
        **kwargs: dict,
    ) -> None:
        super(TransformerBlock, self).__init__(**kwargs)
        # save for import, export, duplication etc
        self._config = {
            'head_num': int(head_num),
            'head_dim': int(head_dim),
            'mlp_dim': int(mlp_dim),
            'causal_opt': bool(causal_opt),}
        # attention sublayer (registered as a submodule immediately)
        self._attention = SelfAttention(
            head_num=head_num,
            head_dim=head_dim,
            causal_opt=causal_opt)
        # build at runtime
        self._norm1_weight = None
        self._norm1_bias = None
        self._norm2_weight = None
        self._norm2_bias = None
        self._w1 = None
        self._b1 = None
        self._w2 = None
        self._b2 = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None,
    ) -> None:
        if not self._built:
            __embed_dim = int(shape[-1])
            __mlp_dim = self._config['mlp_dim']
            # layer norm parameters (pre-attention and pre-FFN)
            self._norm1_weight = torch.nn.Parameter(torch.ones(__embed_dim, device=device, dtype=dtype))
            self._norm1_bias = torch.nn.Parameter(torch.zeros(__embed_dim, device=device, dtype=dtype))
            self._norm2_weight = torch.nn.Parameter(torch.ones(__embed_dim, device=device, dtype=dtype))
            self._norm2_bias = torch.nn.Parameter(torch.zeros(__embed_dim, device=device, dtype=dtype))
            # FFN parameters
            self._w1 = torch.nn.Parameter(torch.empty(__embed_dim, __mlp_dim, device=device, dtype=dtype))
            self._b1 = torch.nn.Parameter(torch.zeros(__mlp_dim, device=device, dtype=dtype))
            self._w2 = torch.nn.Parameter(torch.empty(__mlp_dim, __embed_dim, device=device, dtype=dtype))
            self._b2 = torch.nn.Parameter(torch.zeros(__embed_dim, device=device, dtype=dtype))
            torch.nn.init.xavier_uniform_(self._w1)
            torch.nn.init.xavier_uniform_(self._w2)
            self._built = True

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor=None) -> torch.Tensor:
        __shape = tuple(inputs.shape)
        __embed_dim = __shape[-1]
        self.build(shape=__shape, device=inputs.device, dtype=inputs.dtype)
        # pre-norm + self-attention + residual
        __normed = torch.nn.functional.layer_norm(inputs, [__embed_dim], self._norm1_weight, self._norm1_bias)
        __outputs = inputs + self._attention(__normed, paddings=paddings)
        # pre-norm + feed-forward + residual
        __normed = torch.nn.functional.layer_norm(__outputs, [__embed_dim], self._norm2_weight, self._norm2_bias)
        __ffn = torch.relu(__normed @ self._w1 + self._b1) @ self._w2 + self._b2
        return __outputs + __ffn

    def reset_parameters(self) -> None:
        self._attention.reset_parameters()
        if self._norm1_weight is not None:
            torch.nn.init.ones_(self._norm1_weight)
            torch.nn.init.zeros_(self._norm1_bias)
            torch.nn.init.ones_(self._norm2_weight)
            torch.nn.init.zeros_(self._norm2_bias)
            torch.nn.init.xavier_uniform_(self._w1)
            torch.nn.init.xavier_uniform_(self._w2)

    def output_shape(self, shape: tuple) -> tuple:
        return tuple(shape)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})
