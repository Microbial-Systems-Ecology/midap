# MIDAP - Claude Code Context

## Project Overview

**MIDAP** (Microbial Image Data Analysis Pipeline) is a Python-based automated pipeline for analyzing live-cell microscopy images of bacteria in microfluidics chambers. It handles segmentation, tracking, and lineage analysis for two experimental setups:
- **Family Machine**: Cells grow in clusters/monolayers
- **Mother Machine**: Cells grow in linear microfluidics chambers

Version: 1.2.0 | Python: 3.9–3.x | License: MIT

## Repository Layout

```
midap/                    # Main package
  main.py                 # CLI entry point (run_module)
  config.py               # ConfigParser subclass with validation
  checkpoint.py           # Pipeline restart/checkpoint system
  utils.py                # Logging, GUI helpers, image utils
  main_family_machine.py  # Family machine pipeline orchestration
  main_mother_machine.py  # Mother machine pipeline orchestration
  segmentation/           # Segmentation algorithms (plugin architecture)
  tracking/               # Tracking algorithms (plugin architecture)
  imcut/                  # Image cutout/chamber extraction
  data/                   # DataProcessor, TF pipeline, data reduction
  networks/               # U-Net, DeltaV1, DeltaV2 neural network definitions
  apps/                   # GUI apps (init_GUI.py ~18k lines), download, correction
  correction/             # Napari-based manual correction UI
  midap_jupyter/          # Jupyter notebook integration
tests/                    # pytest test suite mirroring midap/ structure
training/                 # Scripts for training custom models
notebook/                 # Jupyter notebooks for result analysis
euler/                    # Euler HPC cluster setup
settings.ini              # Sample user-editable pipeline configuration
```

## CLI Entry Points

```bash
midap                      # Main pipeline (midap.main:run_module)
correct_segmentation       # Manual post-hoc correction
midap_download             # Download model weights and assets
```

Key `midap` flags:
- `--headless` — no GUI, requires pre-configured settings.ini
- `--headless_cluster` — cluster mode with optional config dir + position ID
- `--restart [PATH]` — resume from checkpoint
- `--create_config` — generate default settings.ini
- `--loglevel 0-7` — verbosity (default 7)
- `--cpu_only` — force CPU execution

## Architecture Patterns

### Plugin System
Segmentation, tracking, and imcut classes are discovered dynamically via `__subclasses__()`. To add a new algorithm:
1. Create a new file in the relevant directory implementing the base class
2. Import it in the `__init__.py` with `from .new_module import *`
3. The config system will automatically detect and validate it

### Base Classes
- `midap/segmentation/base_segmentator.py` — `SegmentationPredictor`
- `midap/tracking/base_tracking.py` — `Tracking`
- `midap/imcut/base_cutout.py` — `CutoutImage`

### Configuration System
`midap/config.py` — `Config` class extending `ConfigParser`:
- Case-sensitive keys (`optionxform = str`)
- Validates available algorithm implementations at load time
- Persisted as `settings.ini` files
- Use `config.write()` / `Config.read()` for file I/O

### Checkpoint / Restart
`midap/checkpoint.py` — saves pipeline state to a checkpoint file so interrupted runs can resume from the last completed step. Handled automatically by the pipeline orchestrators.

### Logging
Use the project's logger utility:
```python
from midap.utils import get_logger
logger = get_logger(name=__file__, logging_level=logging_level)
```
Verbosity controlled by `__VERBOSE` env var (levels 0–7).

## Development Workflow

### Environment Setup
```bash
conda env create -f environment.yml
conda activate midap
pip install -e .
```

For M1/M2 Mac: install `tensorflow-metal` separately.

### Running Tests
```bash
coverage run --source="midap" -m pytest -v tests
coverage report -m
```

Individual test file:
```bash
pytest tests/test_config.py -v
```

### Linting
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=midap/apps/PySimpleGUI.py
```
Only syntax/undefined-name errors are enforced. No line-length limit.

## Key Files to Know

| File | What it does |
|------|-------------|
| `midap/config.py` | Config validation + all settable parameters |
| `midap/main.py` | CLI arg parsing, dispatches to pipeline orchestrators |
| `midap/main_family_machine.py` | Full family machine pipeline steps |
| `midap/main_mother_machine.py` | Full mother machine pipeline steps |
| `midap/apps/init_GUI.py` | Entire PySimpleGUI-based configuration GUI (~18k lines, rarely needs editing) |
| `midap/apps/PySimpleGUI.py` | Vendored third-party GUI lib — do not edit |
| `midap/segmentation/base_segmentator.py` | Base class interface for all segmentors |
| `midap/tracking/base_tracking.py` | Base class interface for all trackers |
| `download_info.json` | Versioned list of downloadable model weights/assets |
| `settings.ini` | Sample configuration file showing all options |

## Segmentation Algorithms

| Class | File | Notes |
|-------|------|-------|
| CellposeSegmentator | `cellpose_sam_segmentator.py` | Cellpose + SAM |
| OmniSegmentator | `omni_segmentator.py` | Omnipose (bacteria-optimized) |
| StarDistSegmentator | `stardist_segmentator.py` | Star-convex polygons |
| UNetSegmentator | `unet_segmentator.py` | Custom U-Net |
| HybridSegmentator | `hybrid_segmentator.py` | Combined approach |

## Tracking Algorithms

| Class | File | Notes |
|-------|------|-------|
| STrackTracking | `strack_tracking.py` | Primary tracker |
| BayesianTracking | `bayesian_tracking.py` | btrack-based |
| DeltaV1Tracking | `deltav1_tracking.py` | Delta neural net |
| DeltaV2Tracking | `deltav2_tracking.py` | Delta v2 |

## Coding Conventions

- **Type hints**: Used throughout; `Union[str, bytes, os.PathLike]` for paths
- **Docstrings**: Google-style with `:param name:` / `:type name:` format
- **No formatters enforced** (black, isort not configured) — match existing style in the file being edited
- **Tests**: Add tests in `tests/<module>/test_<file>.py` mirroring the source tree
- **No breaking changes to settings.ini keys** without updating `Config` validation and the GUI

## CI/CD

Two GitHub Actions workflows:
- `.github/workflows/pytest_with_conda.yml` — primary, uses conda environment
- `.github/workflows/pytest_with_venv.yml` — secondary, venv with Python 3.9 and 3.10

Triggers: push to `development`, PRs, monthly schedule.
Coverage badge updated automatically on `development` branch pushes.

## Common Gotchas

- `midap/apps/PySimpleGUI.py` is vendored (1.8MB) — never edit or lint it
- TensorFlow version differs between standard install (2.13/2.15) and Euler cluster variant
- Model weights are downloaded separately via `midap_download` — tests cache them in CI
- The plugin discovery requires the module to be imported in `__init__.py`
- `Config` uses `optionxform = str` — config keys are case-sensitive
- `pkg_resources` has been replaced (see branch `fix/pkg_resources_deprecated`)
