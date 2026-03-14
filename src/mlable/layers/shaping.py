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

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return mlable.shaping.axes.divide(data=inputs, **self._config)

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

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return mlable.shaping.axes.merge(data=inputs, **self._config)

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

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return mlable.shaping.axes.swap(inputs, **self._config)

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

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return mlable.shaping.axes.move(inputs, **self._config)
