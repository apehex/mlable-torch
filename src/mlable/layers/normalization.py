import torch
import torch.nn
import torch.nn.functional

import mlable.shaping.axes

# GROUP ########################################################################

class GroupNorm(torch.nn.Module):
    def __init__(
        self,
        group_num: int,
        group_axis: int=-1,
        epsilon_val: float=1e-5,
        affine_opt: bool=True,
        **kwargs: dict,
    ) -> None:
        super(GroupNorm, self).__init__(**kwargs)
        # save for import, export, duplication etc
        self._config = {
            'group_num': int(group_num),
            'group_axis': int(group_axis),
            'epsilon_val': float(epsilon_val),
            'affine_opt': bool(affine_opt),}
        # build at runtime
        self._weight = None
        self._bias = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None,
    ) -> None:
        if (not self._built):
            if self._config['affine_opt']:
                # interpret negative axes
                __axis = int(self._config['group_axis'] % len(shape))
                __dim = int(shape[__axis])
                # create empty parameters
                self._weight = torch.nn.Parameter(torch.ones(__dim, dtype=dtype, device=device))
                self._bias = torch.nn.Parameter(torch.zeros(__dim, dtype=dtype, device=device))
            else:
                # reserve the names
                self.register_parameter('_weight', None)
                self.register_parameter('_bias', None)
            # register to avoid the initialization once
            self._built

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        __shape = tuple(inputs.shape)
        __axis = int(self._config['group_axis'] % len(__shape))
        __count = max(1, self._config['group_num'])
        # create the parameters according to the inputs
        self.build(shape=__shape, dtype=inputs.dtype, device=inputs.device)
        # swap the feature axis with axis 1
        __outputs = mlable.shaping.axes.swap(inputs, left_axis=1, right_axis=__axis)
        # `group_norm` expects the features to be on axis 1
        __outputs = torch.nn.functional.group_norm(
            __outputs,
            num_groups=__count,
            weight=self._weight,
            bias=self._bias,
            eps=self._config['epsilon_val'])
        # swap the feature axis back to its original position
        return mlable.shaping.axes.swap(__outputs, left_axis=1, right_axis=__axis)

    def reset_parameters(self) -> None:
        if self._config['affine_opt']:
            torch.nn.init.ones_(self._weight)
            torch.nn.init.zeros_(self._bias)

    def output_shape(self, shape: tuple) -> tuple:
        return tuple(shape)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})
