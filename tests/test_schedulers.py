import math

import pytest
import torch
import torch.nn as nn

from mlable.schedulers import CosineLR, WaveLR

# FIXTURES #####################################################################

@pytest.fixture
def model():
    return nn.Linear(8, 2)

@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.05)

@pytest.fixture
def optimizer_multi(model):
    """Optimizer with two param groups at different LRs."""
    return torch.optim.SGD([
        {'params': [model.weight], 'lr': 0.05},
        {'params': [model.bias], 'lr': 0.5},])

# HELPERS ######################################################################

def cosine_factor(current, total, start, end):
    """Closed-form factor at step t."""
    return (
        0.5 * (start + end)
        + 0.5 * (start - end) * math.cos(math.pi * (current / total)))

def linear_factor(current, total, start, end):
    """Closed-form factor at step t."""
    return start + (end - start) * (min(current, total) / total)

def collect_lrs(scheduler, steps):
    """Step the scheduler and collect group LRs at each step."""
    history = []
    for _ in range(steps):
        history.append([g['lr'] for g in scheduler.optimizer.param_groups])
        scheduler.step()
    return history

# COSINE LR ####################################################################

class TestCosineLR:

    def test_decay_single_group(self, optimizer):
        """Basic cosine decay matches closed-form targets."""
        total, start, end = 10, 1.0, 0.01
        base_lr = 0.05
        scheduler = CosineLR(optimizer, start_rate=start, end_rate=end, total_num=total)
        history = collect_lrs(scheduler, 15)

        for t, (lr,) in enumerate(history, start=1):
            expected = base_lr * cosine_factor(min(t, total), total, start, end)
            assert lr == pytest.approx(expected, rel=1e-9), f"step {t}"

    def test_decay_multi_group(self, optimizer_multi):
        """Factor is applied identically to each param group."""
        total, start, end = 8, 1.0, 0.1
        scheduler = CosineLR(optimizer_multi, start_rate=start, end_rate=end, total_num=total)
        history = collect_lrs(scheduler, 12)

        for t, lrs in enumerate(history, start=1):
            factor = cosine_factor(min(t, total), total, start, end)
            assert lrs[0] == pytest.approx(0.05 * factor, rel=1e-9), f"group 0, step {t}"
            assert lrs[1] == pytest.approx(0.5 * factor, rel=1e-9), f"group 1, step {t}"

    def test_warmup(self, optimizer):
        """Increasing schedule: end_rate > start_rate."""
        total, start, end = 10, 0.1, 1.0
        base_lr = 0.05
        scheduler = CosineLR(optimizer, start_rate=start, end_rate=end, total_num=total)
        history = collect_lrs(scheduler, total)

        # LR should increase monotonically
        lrs = [h[0] for h in history]
        assert all(b >= a for a, b in zip(lrs, lrs[1:])), "LR should be non-decreasing during warmup"

        # final LR should reach base_lr * end_rate
        assert lrs[-1] == pytest.approx(base_lr * end, rel=1e-9)

    def test_constant_after_total(self, optimizer):
        """LR stays constant once total_num is exceeded."""
        total = 5
        scheduler = CosineLR(optimizer, start_rate=1.0, end_rate=0.1, total_num=total)
        history = collect_lrs(scheduler, total + 10)

        lr_at_total = history[total - 1][0]
        for t in range(total, len(history)):
            assert history[t][0] == pytest.approx(lr_at_total, rel=1e-12), f"step {t + 1}"

    def test_endpoints(self, optimizer):
        """Factor equals start_rate at t=0 and end_rate at t=total."""
        total, start, end = 20, 0.5, 0.05
        base_lr = 0.05
        scheduler = CosineLR(optimizer, start_rate=start, end_rate=end, total_num=total)
        history = collect_lrs(scheduler, total)

        assert history[0][0] == pytest.approx(base_lr * start, rel=1e-9), "t=1 (after first step)"
        assert history[-1][0] == pytest.approx(base_lr * end, rel=1e-9), "t=total"

    def test_symmetry(self, optimizer):
        """Cosine is symmetric: factor(k) + factor(total - k) = start + end."""
        total, start, end = 20, 1.0, 0.0
        scheduler = CosineLR(optimizer, start_rate=start, end_rate=end, total_num=total)

        for k in range(total + 1):
            fwd = cosine_factor(k, total, start, end)
            bwd = cosine_factor(total - k, total, start, end)
            assert fwd + bwd == pytest.approx(start + end, rel=1e-12)

    def test_composable_with_exponential(self, optimizer):
        """CosineLR composes correctly with ExponentialLR via compound stepping."""
        total, start, end = 10, 1.0, 0.1
        gamma = 0.9
        base_lr = 0.05

        cosine = CosineLR(optimizer, start_rate=start, end_rate=end, total_num=total)
        exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        for t in range(1, 15):
            cosine.step()
            exp.step()
            expected = (
                base_lr
                * cosine_factor(min(t, total), total, start, end)
                * gamma ** t)
            assert optimizer.param_groups[0]['lr'] == pytest.approx(expected, rel=1e-9), f"step {t}"

    def test_get_last_lr(self, optimizer):
        """get_last_lr matches the optimizer's current LR after each step."""
        scheduler = CosineLR(optimizer, start_rate=0.5, end_rate=0.1, total_num=10)

        for _ in range(15):
            scheduler.step()
            last_lrs = scheduler.get_last_lr()
            actual_lrs = [g['lr'] for g in optimizer.param_groups]
            for last, actual in zip(last_lrs, actual_lrs):
                assert last == pytest.approx(actual, rel=1e-12)

    def test_state_dict_roundtrip(self, optimizer):
        """Save and load produces identical LR sequences."""
        total, start, end = 10, 0.8, 0.05

        # run 5 steps, save
        s1 = CosineLR(optimizer, start_rate=start, end_rate=end, total_num=total)
        collect_lrs(s1, 5)
        state = s1.state_dict()
        lr_after_5 = optimizer.param_groups[0]['lr']

        # continue 5 more steps
        remaining_s1 = collect_lrs(s1, 5)

        # restore and run the same 5 steps
        optimizer.param_groups[0]['lr'] = lr_after_5
        s2 = CosineLR(optimizer, start_rate=start, end_rate=end, total_num=total)
        s2.load_state_dict(state)
        remaining_s2 = collect_lrs(s2, 5)

        for t, (a, b) in enumerate(zip(remaining_s1, remaining_s2)):
            assert a[0] == pytest.approx(b[0], rel=1e-12), f"step {t + 6}"

# WAVE LR ######################################################################

class TestWaveLR:

    def test_warmup_then_decay(self, optimizer):
        """WaveLR increases during warmup, then decays."""
        warmup = 10
        total = 50
        scheduler = WaveLR(optimizer, start_rate=0.01, end_rate=0.01, total_num=total, warmup_num=warmup)
        history = collect_lrs(scheduler, total)
        lrs = [h[0] for h in history]

        # warmup phase: LR should be increasing
        warmup_lrs = lrs[:warmup]
        assert all(b >= a for a, b in zip(warmup_lrs, warmup_lrs[1:])), \
            "LR should be non-decreasing during warmup"

        # decay phase: LR should be decreasing
        decay_lrs = lrs[warmup:]
        assert all(b <= a for a, b in zip(decay_lrs, decay_lrs[1:])), \
            "LR should be non-increasing during decay"

    def test_peak_at_warmup_end(self, optimizer):
        """LR peaks near the warmup-to-decay transition."""
        warmup = 10
        total = 50
        base_lr = 0.05
        scheduler = WaveLR(optimizer, start_rate=0.01, end_rate=0.01, total_num=total, warmup_num=warmup)
        history = collect_lrs(scheduler, total)
        lrs = [h[0] for h in history]

        peak_idx = max(range(len(lrs)), key=lambda i: lrs[i])
        # peak should be at or near the warmup boundary
        assert abs(peak_idx - (warmup - 1)) <= 1, \
            f"Peak at step {peak_idx}, expected near step {warmup - 1}"

        # peak should be close to base_lr (factor ~1.0)
        assert lrs[peak_idx] == pytest.approx(base_lr, rel=0.05)

    def test_warmup_phase_matches_linear(self, optimizer):
        """During warmup, WaveLR should match a standalone LinearLR."""
        warmup = 10
        total = 50
        start_rate = 0.01
        base_lr = 0.05

        scheduler = WaveLR(optimizer, start_rate=start_rate, end_rate=0.01, total_num=total, warmup_num=warmup)
        history = collect_lrs(scheduler, warmup)

        for t, (lr,) in enumerate(history, start=1):
            expected = base_lr * linear_factor(t, warmup, start_rate, 1.0)
            assert lr == pytest.approx(expected, rel=1e-6), f"warmup step {t}"

    def test_end_lr(self, optimizer):
        """At the end of the schedule, LR should reach base_lr * end_rate."""
        warmup = 5
        total = 20
        end_rate = 0.02
        base_lr = 0.05
        scheduler = WaveLR(optimizer, start_rate=0.01, end_rate=end_rate, total_num=total, warmup_num=warmup)
        history = collect_lrs(scheduler, total)

        assert history[-1][0] == pytest.approx(base_lr * end_rate, rel=1e-3)

    def test_multi_group(self, optimizer_multi):
        """WaveLR applies the same schedule proportionally to each param group."""
        warmup = 5
        total = 20
        scheduler = WaveLR(optimizer_multi, start_rate=0.01, end_rate=0.01, total_num=total, warmup_num=warmup)
        history = collect_lrs(scheduler, total)

        for t, lrs in enumerate(history):
            ratio = lrs[1] / lrs[0]
            assert ratio == pytest.approx(10.0, rel=1e-3), \
                f"step {t + 1}: group ratio {ratio} != 10"

    def test_constant_after_total(self, optimizer):
        """LR stays constant once total_num is exceeded."""
        warmup = 5
        total = 15
        scheduler = WaveLR(optimizer, start_rate=0.01, end_rate=0.02, total_num=total, warmup_num=warmup)
        history = collect_lrs(scheduler, total + 10)

        lr_at_total = history[total - 1][0]
        for t in range(total, len(history)):
            assert history[t][0] == pytest.approx(lr_at_total, rel=1e-6), f"step {t + 1}"

    def test_state_dict_roundtrip(self, optimizer):
        """Save and load produces identical LR sequences."""
        warmup = 5
        total = 20
        scheduler = WaveLR(optimizer, start_rate=0.01, end_rate=0.01, total_num=total, warmup_num=warmup)

        # run halfway, save
        mid = total // 2
        collect_lrs(scheduler, mid)
        state = scheduler.state_dict()
        lr_at_mid = optimizer.param_groups[0]['lr']

        # continue
        remaining_original = collect_lrs(scheduler, total - mid)

        # restore
        optimizer.param_groups[0]['lr'] = lr_at_mid
        s2 = WaveLR(optimizer, start_rate=0.01, end_rate=0.01, total_num=total, warmup_num=warmup)
        s2.load_state_dict(state)
        remaining_restored = collect_lrs(s2, total - mid)

        for t, (a, b) in enumerate(zip(remaining_original, remaining_restored)):
            assert a[0] == pytest.approx(b[0], rel=1e-6), f"step {mid + t + 1}"
