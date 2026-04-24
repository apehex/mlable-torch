import math
import pytest
import torch

import mlable.metrics

# META #########################################################################

_B, _T, _V, _H = 2, 8, 64, 16
_K = 5

# FIXTURES #####################################################################

def _logits_from_ranks(ranks: list, vocab_dim: int, shape: tuple) -> torch.Tensor:
    """Build (B, T, V) logits where position 0 has a known top-k ordering."""
    __logits = torch.zeros(shape)
    # assign descending scores to the requested token ids at all positions
    for __rank, __tok in enumerate(ranks):
        __logits[:, :, __tok] = float(len(ranks) - __rank)
    return __logits

# TOPK_RATE ####################################################################

class TestTopkRate:

    def test_rate_one_on_identical_logits_k1(self):
        __x = torch.randn(_B, _T, _V)
        assert mlable.metrics.topk_rate(__x, __x, k_num=1) == pytest.approx(1.0)

    def test_rate_one_on_identical_logits_k(self):
        __x = torch.randn(_B, _T, _V)
        assert mlable.metrics.topk_rate(__x, __x, k_num=_K) == pytest.approx(1.0)

    def test_rate_zero_when_sets_disjoint(self):
        __t = torch.zeros(_B, _T, _V)
        __s = torch.zeros(_B, _T, _V)
        for __i in range(_K):
            __t[:, :, __i] = float(_K - __i)
            __s[:, :, _K + __i] = float(_K - __i)
        assert mlable.metrics.topk_rate(__t, __s, k_num=_K) == pytest.approx(0.0)

    def test_same_set_different_order_is_one(self):
        # same top-k tokens in reverse order -> set comparison gives 1.0
        __tokens = list(range(_K))
        __t = _logits_from_ranks(__tokens, _V, (_B, _T, _V))
        __s = _logits_from_ranks(__tokens[::-1], _V, (_B, _T, _V))
        assert mlable.metrics.topk_rate(__t, __s, k_num=_K) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # k=2, preds top-2 = {0,1}, targets top-2 = {1,2}: overlap fraction = 1/2 = 0.5
        __t = torch.zeros(_B, _T, _V)
        __s = torch.zeros(_B, _T, _V)
        __t[:, :, 1] = 2.0   # target top-1: token 1
        __t[:, :, 2] = 1.0   # target top-2: token 2
        __s[:, :, 0] = 2.0   # pred top-1: token 0
        __s[:, :, 1] = 1.0   # pred top-2: token 1
        # targets={1,2}, preds={0,1}: token 1 is in preds, token 2 is not -> 1/2 = 0.5
        assert mlable.metrics.topk_rate(__t, __s, k_num=2) == pytest.approx(0.5)

    def test_partial_position_match(self):
        # B=1, T=4: first 2 positions match, last 2 do not
        __t = torch.zeros(1, 4, _V)
        __s = torch.zeros(1, 4, _V)
        __t[:, :, 0] = 1.0          # teacher always picks token 0
        __s[:, :2, 0] = 1.0         # student matches first 2
        __s[:, 2:, 1] = 1.0         # student picks different token for last 2
        assert mlable.metrics.topk_rate(__t, __s, k_num=1) == pytest.approx(0.5)

    def test_returns_scalar(self):
        __x = torch.randn(_B, _T, _V)
        __y = mlable.metrics.topk_rate(__x, __x, k_num=_K)
        assert isinstance(__y, torch.Tensor)
        assert len(__y.shape) == 0

    def test_returns_bt_tensor_without_reduction(self):
        __x = torch.randn(_B, _T, _V)
        __y = mlable.metrics.topk_rate(__x, __x, k_num=_K, reduce_opt=False)
        assert __y.shape == (_B, _T)

    def test_none_mask_same_as_ones_mask(self):
        __p = torch.randn(_B, _T, _V)
        __t = torch.randn(_B, _T, _V)
        __m = torch.ones(_B, _T)
        assert mlable.metrics.topk_rate(__p, __t, k_num=_K) == pytest.approx(
            mlable.metrics.topk_rate(__p, __t, k_num=_K, mask_arr=__m).item(), abs=1e-6)

    def test_mask_zeros_out_positions(self):
        # all positions match; mask only first half -> scalar still 1.0
        __x = torch.randn(_B, _T, _V)
        __m = torch.zeros(_B, _T)
        __m[:, :_T // 2] = 1.0
        __r = mlable.metrics.topk_rate(__x, __x, k_num=_K, mask_arr=__m)
        assert __r == pytest.approx(1.0)

    def test_value_in_unit_interval(self):
        __p = torch.randn(_B, _T, _V)
        __t = torch.randn(_B, _T, _V)
        __r = mlable.metrics.topk_rate(__p, __t, k_num=_K)
        assert 0.0 <= float(__r) <= 1.0
