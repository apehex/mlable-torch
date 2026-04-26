import math
import pytest
import torch

import mlable.layers.transformer

# META #########################################################################

_B, _T, _E = 2, 8, 16  # batch, sequence length, embedding dim
_H, _D = 2, 8           # head count, head dim (H*D == E)
_MLP = 32               # MLP hidden dim for TransformerBlock

# SELF ATTENTION ###############################################################

class TestSelfAttention:

    # ── output shape ──────────────────────────────────────────────────────────

    def test_output_shape(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_H, head_dim=_D)
        __x = torch.randn(_B, _T, _E)
        __y = __layer(__x)
        assert tuple(__y.shape) == (_B, _T, _E)

    def test_output_shape_method(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_H, head_dim=_D)
        assert __layer.output_shape((_B, _T, _E)) == (_B, _T, _E)

    # ── no mask ───────────────────────────────────────────────────────────────

    def test_no_padding_valid_finite(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_H, head_dim=_D)
        __x = torch.randn(_B, _T, _E)
        __y = __layer(__x, paddings=None)
        assert tuple(__y.shape) == (_B, _T, _E)
        assert torch.isfinite(__y).all()

    # ── all-zeros padding mask (no padding at all) ────────────────────────────

    def test_zeros_padding_equals_no_mask(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_H, head_dim=_D)
        __layer.eval()
        __x = torch.randn(_B, _T, _E)
        __pad_zeros = torch.zeros(_B, _T)
        with torch.no_grad():
            __out_none = __layer(__x, paddings=None)
            __out_zeros = __layer(__x, paddings=__pad_zeros)
        assert torch.allclose(__out_none, __out_zeros)

    # ── padding mask actually applied ─────────────────────────────────────────

    def test_padding_mask_affects_nonpadding_output(self):
        # use non-causal attention so that early tokens attend to all others
        __layer = mlable.layers.transformer.SelfAttention(head_num=_H, head_dim=_D, causal_opt=False)
        __layer.eval()
        __x = torch.randn(_B, _T, _E)
        # mark the second half of the sequence as padding
        __paddings = torch.zeros(_B, _T)
        __paddings[:, _T // 2:] = 1.0
        with torch.no_grad():
            __out_no_pad = __layer(__x, paddings=None)
            __out_with_pad = __layer(__x, paddings=__paddings)
        # non-padding positions (first half) attended to fewer keys -> output must differ
        assert not torch.allclose(__out_no_pad[:, :_T // 2, :], __out_with_pad[:, :_T // 2, :])

    # ── causal masking ─────────────────────────────────────────────────────────

    def test_causal_future_token_does_not_affect_past(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_H, head_dim=_D, causal_opt=True)
        __layer.eval()
        __x = torch.randn(_B, _T, _E)
        # modify the last token
        __x_mod = __x.clone()
        __x_mod[:, -1, :] = torch.randn(_E)
        with torch.no_grad():
            __out1 = __layer(__x)
            __out2 = __layer(__x_mod)
        # positions 0 … T-2 must be unaffected
        assert torch.allclose(__out1[:, :-1, :], __out2[:, :-1, :])

    def test_causal_all_positions_affected_from_last(self):
        # the last position attends to all positions so changing position 0
        # should affect the output at the last position only when causal=True
        __layer = mlable.layers.transformer.SelfAttention(head_num=_H, head_dim=_D, causal_opt=True)
        __layer.eval()
        __x = torch.randn(_B, _T, _E)
        __x_mod = __x.clone()
        __x_mod[:, 0, :] = torch.randn(_E)
        with torch.no_grad():
            __out1 = __layer(__x)
            __out2 = __layer(__x_mod)
        # all positions from 0 onward can see position 0 as a key, so all outputs differ
        assert not torch.allclose(__out1, __out2)

    # ── determinism ───────────────────────────────────────────────────────────

    def test_deterministic_in_eval_mode(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_H, head_dim=_D)
        __layer.eval()
        __x = torch.randn(_B, _T, _E)
        with torch.no_grad():
            __y1 = __layer(__x)
            __y2 = __layer(__x)
        assert torch.allclose(__y1, __y2)

    # ── dtype / device ────────────────────────────────────────────────────────

    def test_float32_input(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_H, head_dim=_D)
        __x = torch.randn(_B, _T, _E, dtype=torch.float32)
        __y = __layer(__x)
        assert __y.dtype == torch.float32

    # ── config round-trip ─────────────────────────────────────────────────────

    def test_config_roundtrip(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_H, head_dim=_D, causal_opt=False)
        __cfg = __layer.get_config()
        __layer2 = mlable.layers.transformer.SelfAttention.from_config(__cfg)
        assert __layer2.get_config() == __cfg

    def test_get_config_keys(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_H, head_dim=_D)
        __cfg = __layer.get_config()
        assert set(__cfg.keys()) == {'head_num', 'head_dim', 'causal_opt'}

# TRANSFORMER BLOCK ############################################################

class TestTransformerBlock:

    # ── output shape ──────────────────────────────────────────────────────────

    def test_output_shape(self):
        __layer = mlable.layers.transformer.TransformerBlock(head_num=_H, head_dim=_D, mlp_dim=_MLP)
        __x = torch.randn(_B, _T, _E)
        __y = __layer(__x)
        assert tuple(__y.shape) == (_B, _T, _E)

    def test_output_shape_method(self):
        __layer = mlable.layers.transformer.TransformerBlock(head_num=_H, head_dim=_D, mlp_dim=_MLP)
        assert __layer.output_shape((_B, _T, _E)) == (_B, _T, _E)

    # ── basic forward ─────────────────────────────────────────────────────────

    def test_forward_valid_finite(self):
        __layer = mlable.layers.transformer.TransformerBlock(head_num=_H, head_dim=_D, mlp_dim=_MLP)
        __x = torch.randn(_B, _T, _E)
        __y = __layer(__x)
        assert torch.isfinite(__y).all()

    def test_forward_with_padding_valid_finite(self):
        __layer = mlable.layers.transformer.TransformerBlock(head_num=_H, head_dim=_D, mlp_dim=_MLP)
        __x = torch.randn(_B, _T, _E)
        __pad = torch.zeros(_B, _T)
        __pad[:, _T // 2:] = 1.0
        __y = __layer(__x, paddings=__pad)
        assert tuple(__y.shape) == (_B, _T, _E)
        assert torch.isfinite(__y).all()

    # ── padding mask propagated ────────────────────────────────────────────────

    def test_padding_mask_changes_output(self):
        __layer = mlable.layers.transformer.TransformerBlock(head_num=_H, head_dim=_D, mlp_dim=_MLP, causal_opt=False)
        __layer.eval()
        __x = torch.randn(_B, _T, _E)
        __pad = torch.zeros(_B, _T)
        __pad[:, _T // 2:] = 1.0
        with torch.no_grad():
            __out_none = __layer(__x, paddings=None)
            __out_pad = __layer(__x, paddings=__pad)
        assert not torch.allclose(__out_none[:, :_T // 2, :], __out_pad[:, :_T // 2, :])

    # ── determinism ───────────────────────────────────────────────────────────

    def test_deterministic_in_eval_mode(self):
        __layer = mlable.layers.transformer.TransformerBlock(head_num=_H, head_dim=_D, mlp_dim=_MLP)
        __layer.eval()
        __x = torch.randn(_B, _T, _E)
        with torch.no_grad():
            __y1 = __layer(__x)
            __y2 = __layer(__x)
        assert torch.allclose(__y1, __y2)

    # ── dtype / device ────────────────────────────────────────────────────────

    def test_float32_input(self):
        __layer = mlable.layers.transformer.TransformerBlock(head_num=_H, head_dim=_D, mlp_dim=_MLP)
        __x = torch.randn(_B, _T, _E, dtype=torch.float32)
        __y = __layer(__x)
        assert __y.dtype == torch.float32

    # ── config round-trip ─────────────────────────────────────────────────────

    def test_config_roundtrip(self):
        __layer = mlable.layers.transformer.TransformerBlock(head_num=_H, head_dim=_D, mlp_dim=_MLP, causal_opt=False)
        __cfg = __layer.get_config()
        __layer2 = mlable.layers.transformer.TransformerBlock.from_config(__cfg)
        assert __layer2.get_config() == __cfg

    def test_get_config_keys(self):
        __layer = mlable.layers.transformer.TransformerBlock(head_num=_H, head_dim=_D, mlp_dim=_MLP)
        __cfg = __layer.get_config()
        assert set(__cfg.keys()) == {'head_num', 'head_dim', 'mlp_dim', 'causal_opt'}
