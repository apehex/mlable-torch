import math
import pytest
import torch

import mlable.losses

# META #########################################################################

_B, _T, _V, _H = 2, 8, 64, 16

# MSE_LOSS #####################################################################

class TestMseLoss:

    def test_zero_on_identical_inputs(self):
        __x = torch.randn(_B, _T, _H)
        assert mlable.losses.mse_loss(__x, __x) == pytest.approx(0.0, abs=1e-6)

    def test_positive_on_different_inputs(self):
        __p = torch.zeros(_B, _T, _H)
        __t = torch.ones(_B, _T, _H)
        assert mlable.losses.mse_loss(__p, __t).item() > 0.0

    def test_returns_scalar(self):
        __x = torch.randn(_B, _T, _H)
        __y = mlable.losses.mse_loss(__x, __x)
        assert isinstance(__y, torch.Tensor)
        assert len(__y.shape) == 0

    def test_returns_bt_tensor_without_reduction(self):
        __x = torch.randn(_B, _T, _H)
        __y = mlable.losses.mse_loss(__x, __x, reduce_opt=False)
        assert __y.shape == (_B, _T)

    def test_none_mask_same_as_ones_mask(self):
        __p = torch.randn(_B, _T, _H)
        __t = torch.randn(_B, _T, _H)
        __m = torch.ones(_B, _T)
        assert mlable.losses.mse_loss(__p, __t) == pytest.approx(
            mlable.losses.mse_loss(__p, __t, mask_arr=__m).item(), abs=1e-6)

    def test_zeros_mask_gives_zero_scalar(self):
        __p = torch.randn(_B, _T, _H)
        __t = torch.randn(_B, _T, _H)
        __m = torch.zeros(_B, _T)
        assert mlable.losses.mse_loss(__p, __t, mask_arr=__m).item() == pytest.approx(0.0, abs=1e-6)

    def test_zeros_mask_gives_zero_tensor(self):
        __p = torch.randn(_B, _T, _H)
        __t = torch.randn(_B, _T, _H)
        __m = torch.zeros(_B, _T)
        __y = mlable.losses.mse_loss(__p, __t, mask_arr=__m, reduce_opt=False)
        assert __y.shape == (_B, _T)
        assert __y.sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_mask_changes_scalar(self):
        # first T//2 positions: p == t (MSE=0); second T//2: p != t (MSE>0)
        __p = torch.zeros(_B, _T, _H)
        __t = torch.zeros(_B, _T, _H)
        __t[:, _T // 2:, :] = 1.0
        # masking only the high-loss half gives a higher average than the full mean
        __m_all = torch.ones(_B, _T)
        __m_half = torch.zeros(_B, _T)
        __m_half[:, _T // 2:] = 1.0
        __r_all = mlable.losses.mse_loss(__p, __t, mask_arr=__m_all).item()
        __r_half = mlable.losses.mse_loss(__p, __t, mask_arr=__m_half).item()
        assert __r_half > __r_all

    def test_finite_output(self):
        __p = torch.randn(_B, _T, _H)
        __t = torch.randn(_B, _T, _H)
        assert math.isfinite(mlable.losses.mse_loss(__p, __t).item())

# COS_SIM ######################################################################

class TestCosSim:

    def test_one_on_identical_inputs(self):
        __x = torch.randn(_B, _T, _H)
        assert mlable.losses.cos_sim(__x, __x) == pytest.approx(1.0, abs=1e-5)

    def test_zero_on_orthogonal_inputs(self):
        __p = torch.zeros(_B, _T, _H)
        __t = torch.zeros(_B, _T, _H)
        __p[:, :, 0] = 1.0
        __t[:, :, 1] = 1.0
        assert mlable.losses.cos_sim(__p, __t) == pytest.approx(0.0, abs=1e-5)

    def test_returns_scalar(self):
        __x = torch.randn(_B, _T, _H)
        __y = mlable.losses.cos_sim(__x, __x)
        assert isinstance(__y, torch.Tensor)
        assert len(__y.shape) == 0

    def test_returns_bt_tensor_without_reduction(self):
        __x = torch.randn(_B, _T, _H)
        __y = mlable.losses.cos_sim(__x, __x, reduce_opt=False)
        assert __y.shape == (_B, _T)

    def test_none_mask_same_as_ones_mask(self):
        __p = torch.randn(_B, _T, _H)
        __t = torch.randn(_B, _T, _H)
        __m = torch.ones(_B, _T)
        assert mlable.losses.cos_sim(__p, __t) == pytest.approx(
            mlable.losses.cos_sim(__p, __t, mask_arr=__m).item(), abs=1e-6)

    def test_zeros_mask_gives_zero_scalar(self):
        __p = torch.randn(_B, _T, _H)
        __t = torch.randn(_B, _T, _H)
        __m = torch.zeros(_B, _T)
        assert mlable.losses.cos_sim(__p, __t, mask_arr=__m).item() == pytest.approx(0.0, abs=1e-6)

    def test_zeros_mask_gives_zero_tensor(self):
        __p = torch.randn(_B, _T, _H)
        __t = torch.randn(_B, _T, _H)
        __m = torch.zeros(_B, _T)
        __y = mlable.losses.cos_sim(__p, __t, mask_arr=__m, reduce_opt=False)
        assert __y.shape == (_B, _T)
        assert __y.sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_mask_changes_scalar(self):
        # first T//2 positions: identical vectors (cos_sim=1), second T//2: orthogonal (cos_sim=0)
        __p = torch.zeros(_B, _T, _H)
        __t = torch.zeros(_B, _T, _H)
        __p[:, :, 0] = 1.0
        __t[:, :_T // 2, 0] = 1.0           # first half: same vector -> cos_sim=1
        __t[:, _T // 2:, 1] = 1.0           # second half: orthogonal -> cos_sim=0
        # full mean: 0.5; masking only the first half (high similarity) -> mean=1.0
        __m = torch.zeros(_B, _T)
        __m[:, :_T // 2] = 1.0
        __r_all = mlable.losses.cos_sim(__p, __t).item()
        __r_masked = mlable.losses.cos_sim(__p, __t, mask_arr=__m).item()
        assert __r_masked > __r_all

    def test_finite_output(self):
        __p = torch.randn(_B, _T, _H)
        __t = torch.randn(_B, _T, _H)
        assert math.isfinite(mlable.losses.cos_sim(__p, __t).item())

# KL_DIVERGENCE ################################################################

class TestKlDiv:

    def test_zero_on_identical_logits(self):
        __x = torch.randn(_B, _T, _V)
        assert mlable.losses.kl_div(__x, __x) == pytest.approx(0.0, abs=1e-4)

    def test_positive_on_different_logits(self):
        __t = torch.zeros(_B, _T, _V)
        __s = torch.zeros(_B, _T, _V)
        __t[:, :, 0] = 10.0
        __s[:, :, 1] = 10.0
        assert mlable.losses.kl_div(__t, __s).item() > 0.0

    def test_returns_scalar(self):
        __x = torch.randn(_B, _T, _V)
        __y = mlable.losses.kl_div(__x, __x)
        assert isinstance(__y, torch.Tensor)
        assert len(__y.shape) == 0

    def test_returns_bt_tensor_without_reduction(self):
        __x = torch.randn(_B, _T, _V)
        __y = mlable.losses.kl_div(__x, __x, reduce_opt=False)
        assert __y.shape == (_B, _T)

    def test_none_mask_same_as_ones_mask(self):
        __p = torch.randn(_B, _T, _V)
        __t = torch.randn(_B, _T, _V)
        __m = torch.ones(_B, _T)
        assert mlable.losses.kl_div(__p, __t) == pytest.approx(
            mlable.losses.kl_div(__p, __t, mask_arr=__m).item(), abs=1e-6)

    def test_zeros_mask_gives_zero_scalar(self):
        __p = torch.randn(_B, _T, _V)
        __t = torch.randn(_B, _T, _V)
        __m = torch.zeros(_B, _T)
        assert mlable.losses.kl_div(__p, __t, mask_arr=__m).item() == pytest.approx(0.0, abs=1e-6)

    def test_zeros_mask_gives_zero_tensor(self):
        __p = torch.randn(_B, _T, _V)
        __t = torch.randn(_B, _T, _V)
        __m = torch.zeros(_B, _T)
        __y = mlable.losses.kl_div(__p, __t, mask_arr=__m, reduce_opt=False)
        assert __y.shape == (_B, _T)
        assert __y.sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_mask_changes_scalar(self):
        # first T//2 positions: identical (KL=0), second T//2: different (KL>0)
        __p = torch.zeros(_B, _T, _V)
        __t = torch.zeros(_B, _T, _V)
        __p[:, :, 0] = 10.0
        __t[:, :_T // 2, 0] = 10.0          # first half: same -> KL=0
        __t[:, _T // 2:, 1] = 10.0          # second half: different -> KL>0
        # masking only the high-KL half gives a higher average than the full mean
        __m = torch.zeros(_B, _T)
        __m[:, _T // 2:] = 1.0
        __r_all = mlable.losses.kl_div(__p, __t).item()
        __r_masked = mlable.losses.kl_div(__p, __t, mask_arr=__m).item()
        assert __r_masked > __r_all

    def test_finite_output(self):
        __p = torch.randn(_B, _T, _V)
        __t = torch.randn(_B, _T, _V)
        assert math.isfinite(mlable.losses.kl_div(__p, __t).item())
