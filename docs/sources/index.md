# Sources Index

Overview of the `mlable` package modules and their public symbols.

## Top-level Modules

| Module           | Purpose                        | Public Symbols                                                         |
| ---------------- | ------------------------------ | ---------------------------------------------------------------------- |
| `losses.py`      | Loss functions                 | `mse_loss`, `cos_sim`, `kl_div`                                        |
| `metrics.py`     | Metric functions               | `topk_rate`                                                            |
| `models.py`      | Model building blocks          | `freeze`, `free_memory`                                                |
| `schedulers.py`  | LR schedulers                  | `CosineLR`, `WaveLR`                                                   |
| `shapes.py`      | Shape arithmetic helpers       | `normalize_dim`, `symbolic_dim`, `multiply_dim`, `divide_dim`, `normalize`, `symbolic`, `filter`, `divide`, `merge`, `swap`, `move` |
| `utils.py`       | General utilities              | `chunk`, `merge`, `rotate`, `logroot2`, `exproot2`, `ema`, `iterable`  |

## Sub-modules

### `encoding/`

Encoding helpers for converting raw data to tensor-ready formats.

| Module           | Purpose                        | Public Symbols                                                                              |
| ---------------- | ------------------------------ | ------------------------------------------------------------------------------------------- |
| `rgb.py`         | RGB encoding utilities         | `clean`, `split`, `pad`, `rgb_utf`, `mix_channels`, `rgb_mixed`, `rgb_hilbert`, `restore`, `decode` |

### `layers/`

Standalone `torch.nn.Module` layers.

| Module             | Purpose                  | Public Symbols                                   |
| ------------------ | ------------------------ | ------------------------------------------------ |
| `embedding.py`     | Embedding layers         | `PositionalEmbedding`, `CompositeEmbedding`      |
| `normalization.py` | Normalization layers     | `GroupNorm`                                      |
| `shaping.py`       | Shaping layers           | `Divide`, `Merge`, `Swap`, `Move`                |

### `shaping/`

Functional axis and spatial transforms operating on tensors.

| Module        | Purpose                       | Public Symbols                        |
| ------------- | ----------------------------- | ------------------------------------- |
| `axes.py`     | Axis manipulation helpers     | `divide`, `merge`, `swap`, `move`     |
| `hilbert.py`  | Hilbert curve transforms      | `permutation`, `fold`, `unfold`       |
