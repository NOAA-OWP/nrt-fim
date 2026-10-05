# NRT-FIM CNN

<p align="center">
    <img src="docs/img/f1.png" alt="icefabric" width="25%"/>
</p>

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A repo for the MVP NRT CNN Capability. This work builds on the prototype model trained on MODIS flood satellite images


Getting Started

This repo is managed through UV and can be installed through:
uv venv .venv --python 3.12.0
source .venv/bin/activate
uv sync                 # installs everything from pyproject.toml
uv pip install -e .      # editable install of your own package

## Development

To ensure that F1-trainer code changes follow the specified structure, be sure to install the local dev dependencies and run pre-commit install

## Documentation

To build the user guide documentation for F1-trainer locally, run the following commands:

uv pip install ".[docs]"
mkdocs serve -a localhost:8080
Docs will be spun up at localhost:8080/


## How to run:
### 1- Pre-processing the target data:

    python -m scripts.data_prep.preprocess_modis_targets

This will:
discover raw MODIS TIFFs under ${data_sources.dfo_modis_dir}/DFO_* / *.tif,
(optionally) create a master grid (from precip) and pass it via --grid,
generate the flood-percent + regridded TIFFs (same folder layout you had),
write a manifest file listing all produced outputs (default: data/indices/modis_preprocessed.txt).

### 2- Build index:

    python -m scripts.data_prep.make_index 

### 3- Write splits:

    python -m scripts.data_prep.make_splits

### 4- Compute stats:

    python -m scripts.data_prep.compute_stats

### 5- Train:

    python -m scripts.make_train

### 6- Evaluation:

    python -m scripts.make_eval
