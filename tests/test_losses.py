import math
import pytest
import torch

import mlable.losses

# META #########################################################################

_B, _T, _V = 2, 8, 64  # small synthetic shapes
_K = 5

# KL_DIVERGENCE ################################################################

class TestKlDivergence:

    def test_zero_on_identical_logits(self):
        __x = torch.randn(_B, _T, _V)
        __kl = mlable.losses.kl_divergence(__x, __x).item()
        assert __kl == pytest.approx(0.0, abs=1e-4)

    def test_positive_on_different_logits(self):
        __t = torch.zeros(_B, _T, _V)
        __s = torch.zeros(_B, _T, _V)
        __t[:, :, 0] = 10.0  # teacher strongly prefers token 0
        __s[:, :, 1] = 10.0  # student strongly prefers token 1
        assert mlable.losses.kl_divergence(__t, __s).item() > 0.0

    def test_returns_scalar(self):
        __x = torch.randn(_B, _T, _V)
        __y = mlable.losses.kl_divergence(__x, __x)
        assert isinstance(__y, torch.Tensor)
        assert len(__y.shape) == 0

    def test_finite_output(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __kl = mlable.losses.kl_divergence(__t, __s).item()
        assert math.isfinite(__kl)
