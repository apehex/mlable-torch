import math
import pytest
import torch

import mlable.metrics

# META #########################################################################

_B, _T, _V = 2, 8, 64  # small synthetic shapes
_K = 5

# FIXTURES #####################################################################

def _logits_from_ranks(ranks: list, vocab_dim: int, shape: tuple) -> torch.Tensor:
    """Build (B, T, V) logits where position 0 has a known top-k ordering."""
    __logits = torch.zeros(shape)
    # assign descending scores to the requested token ids at all positions
    for __rank, __tok in enumerate(ranks):
        __logits[:, :, __tok] = float(len(ranks) - __rank)
    return __logits

# TOP1_MATCH_RATE ##############################################################

class TestTop1MatchRate:

    def test_rate_one_on_identical_logits(self):
        __x = torch.randn(_B, _T, _V)
        assert mlable.metrics.topk_rate(__x, __x, k_num=1) == pytest.approx(1.0)

    def test_rate_zero_when_all_mismatch(self):
        # teacher always picks token 0, student always picks token 1
        __t = torch.zeros(_B, _T, _V)
        __t[:, :, 0] = 1.0
        __s = torch.zeros(_B, _T, _V)
        __s[:, :, 1] = 1.0
        assert mlable.metrics.topk_rate(__t, __s, k_num=1) == pytest.approx(0.0)

    def test_partial_match(self):
        # B=1, T=4: first 2 positions match, last 2 do not
        __t = torch.zeros(1, 4, _V)
        __s = torch.zeros(1, 4, _V)
        __t[:, :, 0] = 1.0          # teacher always picks 0
        __s[:, :2, 0] = 1.0         # student picks 0 for first 2
        __s[:, 2:, 1] = 1.0         # student picks 1 for last 2
        assert mlable.metrics.topk_rate(__t, __s, k_num=1) == pytest.approx(0.5)

    def test_returns_scalar(self):
        __x = torch.randn(_B, _T, _V)
        __y = mlable.metrics.topk_rate(__x, __x, k_num=1)
        assert isinstance(__y, torch.Tensor)
        assert len(__y.shape) == 0

    def test_value_in_unit_interval(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __r = mlable.metrics.topk_rate(__t, __s, k_num=1)
        assert 0.0 <= __r <= 1.0

# TOPK MATCH RATE ##############################################################

class TestTopkOrderMatchRate:

    def test_rate_one_on_identical_logits(self):
        __x = torch.randn(_B, _T, _V)
        assert mlable.metrics.topk_rate(__x, __x, k_num=_K) == pytest.approx(1.0)

    def test_rate_zero_when_sets_disjoint(self):
        __t = torch.zeros(_B, _T, _V)
        __s = torch.zeros(_B, _T, _V)
        for __i in range(_K):
            __t[:, :, __i] = float(_K - __i)
            __s[:, :, _K + __i] = float(_K - __i)
        assert mlable.metrics.topk_rate(__t, __s, k_num=_K) == pytest.approx(0.0)

    def test_same_set_different_order_is_zero(self):
        # same top-k tokens but in reverse order -> exact order match fails
        __tokens = list(range(_K))
        __t = _logits_from_ranks(__tokens, _V, (_B, _T, _V))
        __s = _logits_from_ranks(__tokens[::-1], _V, (_B, _T, _V))
        # only matches if the reversed order happens to be the same (k_num=1 edge case excluded)
        if _K > 1:
            assert mlable.metrics.topk_rate(__t, __s, k_num=_K) == pytest.approx(0.0)

    def test_returns_scalar(self):
        __x = torch.randn(_B, _T, _V)
        __y = mlable.metrics.topk_rate(__x, __x, k_num=_K)
        assert isinstance(__y, torch.Tensor)
        assert len(__y.shape) == 0

    def test_value_in_unit_interval(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __r = mlable.metrics.topk_rate(__t, __s, k_num=_K)
        assert 0.0 <= __r <= 1.0
