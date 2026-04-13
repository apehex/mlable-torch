import torch
import torch.nn

import mlable.shaping.axes

# DIVIDE #######################################################################

class Divide(torch.nn.Module):
    def __init__(
        self,
        axis: int, # relative to the original shape
        factor: int,
        insert: bool=False,
        right: bool=True,
        **kwargs
    ) -> None:
        super(Divide, self).__init__(**kwargs)
        # save for import, export, duplication, etc
        self._config = {
            'axis': axis,
            'factor': factor,
            'insert': insert,
            'right': right,}
        # register
        self._built = True

    def build(self, shape: tuple=(), device: object=None, dtype: object=None) -> None:
        self._built = True

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return mlable.shaping.axes.divide(data=inputs, **self._config)

    def reset_parameters(self) -> None:
        return None

    def output_shape(self, shape: tuple) -> tuple:
        return tuple(mlable.shapes.divide(shape, **self._config))

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})

# MERGE ########################################################################

class Merge(torch.nn.Module):
    def __init__(
        self,
        axis: int,
        right: bool=True,
        **kwargs
    ) -> None:
        super(Merge, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'axis': axis,
            'right': right,}
        # register
        self._built = True

    def build(self, shape: tuple=(), device: object=None, dtype: object=None) -> None:
        self._built = True

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return mlable.shaping.axes.merge(data=inputs, **self._config)

    def reset_parameters(self) -> None:
        return None

    def output_shape(self, shape: tuple) -> tuple:
        return tuple(mlable.shapes.merge(shape, **self._config))

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})

# SWAP #########################################################################

class Swap(torch.nn.Module):
    def __init__(
        self,
        left_axis: int,
        right_axis: int,
        **kwargs
    ) -> None:
        super(Swap, self).__init__(**kwargs)
        # save for import / export
        self._config = {'left_axis': left_axis, 'right_axis': right_axis,}
        # register
        self._built = True

    def build(self, shape: tuple=(), device: object=None, dtype: object=None) -> None:
        self._built = True

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return mlable.shaping.axes.swap(inputs, **self._config)

    def reset_parameters(self) -> None:
        return None

    def output_shape(self, shape: tuple) -> tuple:
        return tuple(mlable.shapes.swap(shape, left=self._config['left_axis'], right=self._config['right_axis']))

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})

# MOVE #########################################################################

class Move(torch.nn.Module):
    def __init__(
        self,
        from_axis: int,
        to_axis: int,
        **kwargs
    ) -> None:
        super(Move, self).__init__(**kwargs)
        # save for import / export
        self._config = {'from_axis': from_axis, 'to_axis': to_axis,}
        # register
        self._built = True

    def build(self, shape: tuple=(), device: object=None, dtype: object=None) -> None:
        self._built = True

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return mlable.shaping.axes.move(inputs, **self._config)

    def reset_parameters(self) -> None:
        return None

    def output_shape(self, shape: tuple) -> tuple:
        return tuple(mlable.shapes.move(shape, before=self._config['from_axis'], after=self._config['to_axis']))

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})
