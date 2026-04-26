import math
import pytest
import torch

import mlable.layers.transformer

# META #########################################################################

_B, _T, _H = 2, 8, 32
_HIDDEN, _OUT = 64, 16
_HEADS = 4

# GATED LINEAR UNIT ############################################################

class TestGatedLinearUnit:

    def test_init_stores_config(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        assert __layer._config['hidden_dim'] == _HIDDEN
        assert __layer._config['output_dim'] == _OUT

    def test_not_built_before_forward(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        assert not __layer._built
        assert __layer._extend is None
        assert __layer._project is None

    def test_built_after_forward(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        __layer(torch.randn(_B, _T, _H))
        assert __layer._built
        assert __layer._extend is not None
        assert __layer._project is not None

    def test_build_is_idempotent(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        __layer(torch.randn(_B, _T, _H))
        __extend_first = __layer._extend
        __layer(torch.randn(_B, _T, _H))
        assert __layer._extend is __extend_first

    def test_forward_rank3_output_shape(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        __y = __layer(torch.randn(_B, _T, _H))
        assert tuple(__y.shape) == (_B, _T, _OUT)

    def test_forward_rank2_output_shape(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        __y = __layer(torch.randn(_B, _H))
        assert tuple(__y.shape) == (_B, _OUT)

    def test_forward_rank4_output_shape(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        __y = __layer(torch.randn(_B, _T, _T, _H))
        assert tuple(__y.shape) == (_B, _T, _T, _OUT)

    def test_output_shape_method_rank3(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        assert __layer.output_shape((_B, _T, _H)) == (_B, _T, _OUT)

    def test_output_shape_method_rank2(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        assert __layer.output_shape((_B, _H)) == (_B, _OUT)

    def test_output_shape_matches_forward(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        __x = torch.randn(_B, _T, _H)
        __y = __layer(__x)
        assert tuple(__y.shape) == __layer.output_shape(tuple(__x.shape))

    def test_output_dtype_preserved_float32(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        __x = torch.randn(_B, _T, _H, dtype=torch.float32)
        assert __layer(__x).dtype == torch.float32

    def test_output_is_finite(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        __y = __layer(torch.randn(_B, _T, _H))
        assert torch.isfinite(__y).all()

    def test_get_config_contains_hidden_dim(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        assert __layer.get_config()['hidden_dim'] == _HIDDEN

    def test_get_config_contains_output_dim(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        assert __layer.get_config()['output_dim'] == _OUT

    def test_get_config_is_copy(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        __config = __layer.get_config()
        __config['hidden_dim'] = 999
        assert __layer._config['hidden_dim'] == _HIDDEN

    def test_from_config_roundtrip(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        __layer2 = mlable.layers.transformer.GatedLinearUnit.from_config(__layer.get_config())
        assert __layer2._config['hidden_dim'] == _HIDDEN
        assert __layer2._config['output_dim'] == _OUT

    def test_from_config_produces_working_layer(self):
        __layer = mlable.layers.transformer.GatedLinearUnit(hidden_dim=_HIDDEN, output_dim=_OUT)
        __layer2 = mlable.layers.transformer.GatedLinearUnit.from_config(__layer.get_config())
        __y = __layer2(torch.randn(_B, _T, _H))
        assert tuple(__y.shape) == (_B, _T, _OUT)

# SELF-ATTENTION ###############################################################

class TestSelfAttention:

    def test_init_stores_head_num(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        assert __layer._config['head_num'] == _HEADS

    def test_init_stores_attention_idx(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS, attention_idx=-3)
        assert __layer._config['attention_idx'] == -3

    def test_init_default_attention_idx(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        assert __layer._config['attention_idx'] == -2

    def test_init_stores_bias_opt(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS, bias_opt=False)
        assert __layer._config['bias_opt'] is False

    def test_init_stores_dropout_rate(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS, dropout_rate=0.1)
        assert __layer._config['dropout_rate'] == pytest.approx(0.1)

    def test_not_built_before_forward(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        assert not __layer._built
        assert __layer._layer is None

    def test_built_after_forward(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __layer(torch.randn(_B, _T, _H))
        assert __layer._built
        assert __layer._layer is not None

    def test_build_is_idempotent(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __layer(torch.randn(_B, _T, _H))
        __inner = __layer._layer
        __layer(torch.randn(_B, _T, _H))
        assert __layer._layer is __inner

    def test_forward_output_shape_matches_input(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __x = torch.randn(_B, _T, _H)
        assert tuple(__layer(__x).shape) == (_B, _T, _H)

    def test_output_shape_method_returns_same_shape(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        assert __layer.output_shape((_B, _T, _H)) == (_B, _T, _H)

    def test_output_shape_matches_forward(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __x = torch.randn(_B, _T, _H)
        __y = __layer(__x)
        assert tuple(__y.shape) == __layer.output_shape(tuple(__x.shape))

    def test_forward_rank4_output_shape(self):
        # (B, T1, T2, F) with default attention_idx=-2 attends over T2
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __x = torch.randn(_B, 4, _T, _H)
        assert tuple(__layer(__x).shape) == (_B, 4, _T, _H)

    def test_custom_attention_idx(self):
        # (B, T1, T2, F) with attention_idx=-3 attends over T1
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS, attention_idx=-3)
        __x = torch.randn(_B, _T, 4, _H)
        assert tuple(__layer(__x).shape) == (_B, _T, 4, _H)

    def test_forward_with_none_padding_same_as_no_padding(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __x = torch.randn(_B, _T, _H)
        torch.manual_seed(0)
        __y1 = __layer(__x)
        torch.manual_seed(0)
        __y2 = __layer(__x, paddings=None)
        assert torch.allclose(__y1, __y2)

    def test_unpadded_positions_not_all_zero(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __x = torch.randn(_B, _T, _H)
        # only last 2 positions are padded
        __mask = torch.zeros(_B, _T)
        __mask[:, _T - 2:] = 1.0
        __y = __layer(__x, paddings=__mask)
        assert __y[:, :_T - 2, :].abs().sum() > 0

    def test_all_positions_unpadded_mask_same_as_no_mask(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __x = torch.randn(_B, _T, _H)
        __mask = torch.zeros(_B, _T)  # all zeros = no padding
        __y1 = __layer(__x)
        __y2 = __layer(__x, paddings=__mask)
        assert torch.allclose(__y1, __y2)

    def test_rank_assertion_raised_for_rank2(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        with pytest.raises(AssertionError):
            __layer(torch.randn(_B, _H))

    def test_padding_shape_mismatch_raises(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __x = torch.randn(_B, _T, _H)
        __mask = torch.zeros(_B, _T - 1)  # wrong sequence length
        with pytest.raises(AssertionError):
            __layer(__x, paddings=__mask)

    def test_output_dtype_preserved_float32(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __x = torch.randn(_B, _T, _H, dtype=torch.float32)
        assert __layer(__x).dtype == torch.float32

    def test_output_is_finite(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __y = __layer(torch.randn(_B, _T, _H))
        assert torch.isfinite(__y).all()

    def test_get_config_contains_head_num(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        assert __layer.get_config()['head_num'] == _HEADS

    def test_get_config_contains_attention_idx(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS, attention_idx=-3)
        assert __layer.get_config()['attention_idx'] == -3

    def test_get_config_contains_bias_opt(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS, bias_opt=False)
        assert __layer.get_config()['bias_opt'] is False

    def test_get_config_contains_dropout_rate(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS, dropout_rate=0.1)
        assert __layer.get_config()['dropout_rate'] == pytest.approx(0.1)

    def test_get_config_is_copy(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __config = __layer.get_config()
        __config['head_num'] = 999
        assert __layer._config['head_num'] == _HEADS

    def test_from_config_roundtrip(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS, attention_idx=-3, bias_opt=False)
        __cfg = __layer.get_config()
        __layer2 = mlable.layers.transformer.SelfAttention.from_config(__cfg)
        assert __layer2._config['head_num'] == _HEADS
        assert __layer2._config['attention_idx'] == -3
        assert __layer2._config['bias_opt'] is False

    def test_from_config_produces_working_layer(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        __layer2 = mlable.layers.transformer.SelfAttention.from_config(__layer.get_config())
        __y = __layer2(torch.randn(_B, _T, _H))
        assert tuple(__y.shape) == (_B, _T, _H)

    def test_swapped_shape_default_idx(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        # default attention_idx=-2 swapped with -2 => same shape
        assert tuple(__layer.swapped_shape((_B, _T, _H))) == (_B, _T, _H)

    def test_swapped_shape_custom_idx(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS, attention_idx=-3)
        # (B, T1, T2, F) with attention_idx=-3: swap(-3, -2) swaps T1 and T2
        assert tuple(__layer.swapped_shape((_B, _T, 4, _H))) == (_B, 4, _T, _H)

    def test_merged_shape(self):
        __layer = mlable.layers.transformer.SelfAttention(head_num=_HEADS)
        # (B, T1, T2, F) -> (B*T1, T2, F)
        assert __layer.merged_shape((_B, 4, _T, _H)) == (_B * 4, _T, _H)
