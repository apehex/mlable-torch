# Index

Overview of the repository structure.

## Docs

| Path                      | Purpose                               |
| ------------------------- | ------------------------------------- |
| `-- docs/`                | Collaboration-focused project docs    |
| `   -- agents.md`         | Guidelines for LLM agents             |
| `   -- context.md`        | Overview of the project               |
| `   -- decisions.md`      | Record of important design choices    |
| `   -- index.md`          | Structure of the repository           |
| `   -- invariants.md`     | Hard constraints                      |
| `   -- references.md`     | External references                   |
| `   -- roadmap.md`        | Planning of the project               |

## Sources

| Path                              | Purpose                                       |
| --------------------------------- | --------------------------------------------- |
| `-- src/mlable/`                  | Root of the Python package                    |
| `   -- encoding/`                 | Encoding helpers                              |
| `      -- rgb.py`                 | RGB encoding utilities                        |
| `   -- layers/`                   | Standalone `torch.nn.Module` layers           |
| `      -- embedding.py`           | Embedding layers                              |
| `      -- normalization.py`       | Normalization layers                          |
| `      -- shaping.py`             | Shaping layers                                |
| `   -- shaping/`                  | Axis and spatial transforms                   |
| `      -- axes.py`                | Axis manipulation helpers                     |
| `      -- hilbert.py`             | Hilbert curve transforms                      |
| `   -- losses.py`                 | Loss functions                                |
| `   -- metrics.py`                | Metric functions                              |
| `   -- models.py`                 | Model building blocks                         |
| `   -- schedulers.py`             | LR schedulers                                 |
| `   -- shapes.py`                 | Shape arithmetic helpers                      |
| `   -- utils.py`                  | General utilities                             |

## Tests

| Path                              | Purpose                               |
| --------------------------------- | ------------------------------------- |
| `-- tests/`                       | Unit tests mirroring package layout   |
| `   -- test_losses.py`            | Tests for `losses.py`                 |
| `   -- test_metrics.py`           | Tests for `metrics.py`                |
| `   -- test_schedulers.py`        | Tests for `schedulers.py`             |
