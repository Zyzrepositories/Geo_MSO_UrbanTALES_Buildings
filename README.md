# Geo-MSO: UrbanTALES wind-field surrogate

PyTorch research code for fast pedestrian-level urban wind-field prediction from
building geometry and flow forcing. The repository contains the reusable model,
data-loading, training, evaluation, metric and test code used for a
geometry-conditioned multi-scale surrogate study on UrbanTALES.

## Scope of this release

Included:

- `src/urbantales_ml/`: data catalogue, preprocessing, models, losses, metrics,
  training and full-field inference;
- `scripts/`: core data-audit, training, evaluation, aggregation and plotting
  commands;
- `tests/`: unit tests for data semantics, models, metrics and frozen protocols;
- minimal smoke-test, split and evaluation protocol configurations;
- reproducible Python and AutoDL CUDA 11.8 environment specifications.

Not included: UrbanTALES raw data, prediction arrays, training logs, server
archives, or model checkpoints. This keeps the Git repository small and prevents
the dataset license from being confused with the software license.

## Data

Download the Idealized Building Blocks and Realistic Urban Neighbourhoods from
the [UrbanTALES portal](https://urbantales.vercel.app/), then place the two folders
at the repository root without renaming them:

```text
urbantales-geo-mso/
  Idealized Building Blocks/
  Realistic Urban Neighbourhoods/
```

Place the portal's `metadata.csv` at the repository root as well. Generate the
two small local audit products required by the integration tests with:

```bash
python scripts/audit_urbantales.py --root . --output reports/data_audit
python scripts/resolve_release_semantics.py
```

Those directories are ignored by Git. See [DATA_LICENSE.md](DATA_LICENSE.md) for
the data attribution and CC BY-NC 4.0 notice.

## Environment

The formal server environment targets Python 3.10, PyTorch 2.5.1 and CUDA 11.8:

```bash
conda env create -f environment-autodl.yml
conda activate urbantales-ml
pip install -e . --no-deps
```

For a lightweight local setup, install the packages in
`requirements-local-smoke.txt` plus a PyTorch build suitable for the local host,
then install this package in editable mode.

## Verify the release

```bash
python scripts/validate_splits.py
python -m pytest -q
```

Tests that load real cases are skipped until the UrbanTALES folders, official
metadata and generated audit products above are present.

## Smoke training

The smoke configuration reads real UrbanTALES NetCDF/topography inputs and runs
only two training and two validation steps. It is a pipeline test, not an
accuracy experiment.

```bash
python scripts/run_experiment.py --config configs/experiments/smoke_local.yaml
```

## Full training and evaluation

Use a YAML file with the same schema as the smoke configuration, increasing the
case limits, patch size, model width and epoch count for formal training:

```bash
python scripts/run_experiment.py --config path/to/experiment.yaml
python scripts/evaluate_full_fields.py \
  --config path/to/experiment.yaml \
  --checkpoint runs/example/best.pt \
  --partition val \
  --output runs/example_fullfield_val
```

The evaluation code reconstructs complete periodic fields and reports global,
component-wise, near-building, high-gradient and physics-related diagnostics.
Latency benchmarking is implemented in `scripts/benchmark_latency.py`; its
formal mode validates the locked hardware, configuration and checkpoint hashes.

## Reproducibility notes

- Random seeds and deterministic mode are controlled in YAML configuration.
- UrbanTALES velocity variables are normalized by case friction velocity;
  second-order quantities use the squared scale.
- `FLUX-*` forcing is resolved from each case's physical forcing vector rather
  than guessed from a directory suffix.
- `Val-*` names denote paired spatial-resolution cases and are not ML validation
  labels.
- Training and evaluation outputs are written under `runs/`, which is ignored by
  Git.

## License

Original software in this repository is released under the MIT License. The
UrbanTALES data have separate CC BY-NC 4.0 terms; see `DATA_LICENSE.md`.
