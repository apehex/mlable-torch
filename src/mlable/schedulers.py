import math

import torch
import torch.optim.lr_scheduler

# COSINE #######################################################################

class CosineLR(torch.optim.lr_scheduler.LRScheduler):

    def __init__(
        self,
        optimizer_obj: object,
        start_rate: float=1.0,
        end_rate: float=0.01,
        total_num: int=128,
        current_num: int=-1,
        epsilon_val: float=1e-5,
    ) -> None:
        # save for import, export, duplication etc
        self._config = {
            'start_rate': max(epsilon_val, float(start_rate)),
            'end_rate': max(epsilon_val, float(end_rate)),
            'total_num': max(1, int(total_num)),
            'current_num': max(0, int(current_num)),
            'epsilon_val': epsilon_val,}
        # compute the initial LR
        super().__init__(optimizer_obj, current_num)

    @override
    def get_lr(self) -> list[float | Tensor]:
        """Compute the next learning rate for each of the optimizer_obj's groups."""
        _warn_get_lr_called_within_step(self)
        # (T-1) is not defined when T is zero
        if self.last_epoch == 0:
            return [
                __g["lr"] * self._config['start_rate']
                for __g in self.optimizer.param_groups]
        # keep the LR constant once the iteration counter exceeds the total
        if self._is_initial or (self.last_epoch > self._config['total_num']):
            return _param_groups_val_list(self.optimizer, "lr")
        # 0 < T < T_e
        return [
            __g["lr"] * (
                self._compute_rate(self.last_epoch)
                / self._compute_rate(self.last_epoch - 1))
            for __g in self.optimizer.param_groups]

    def _compute_rate(self, iter_num: int) -> float:
        return (
            0.5 * (self._config['start_rate'] + self._config['end_rate'])
            + 0.5 * (self._config['start_rate'] - self._config['end_rate'])
            * math.cos(math.pi * (iter_num / self._config['total_num'])))

# WAVE #########################################################################

class WaveLR(torch.optim.lr_scheduler.SequentialLR):

    def __init__(
        self,
        optimizer_obj: object,
        start_rate: float=0.0001,
        end_rate: float=0.01,
        total_num: int=128,
        warmup_num: int=-1,
    ) -> None:
        # linear warmup from start factor to 1
        __warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer_obj,
            start_factor=start_rate,
            end_factor=1.0,
            total_iters=warmup_num)
        # cosine decay from 1 to end factor
        __decay = CosineLR(
            optimizer_obj=optimizer_obj,
            start_rate=1.0,
            end_rate=end_rate,
            total_num=max(1, total_num - warmup_num))
        # put the 2 schedulers one after another
        super().__init__(
            optimizer=optimizer_obj,
            schedulers=[__warmup, __decay],
            milestones=[warmup_num],)
