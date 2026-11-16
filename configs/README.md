# Configs

- `baseline.json` holds the shared defaults (data paths, ConvNeXt-T, optimiser, schedule, evaluation).
- `experiments/*.json` are generated from it by `python scripts/make_configs.py`; each experiment changes only
  the components under study (`meta_fusion`, `film_stages`, `loss`, `tau`, `meta_dropout`, `train_fraction`,
  backbone). Do not edit them by hand; edit `scripts/make_configs.py` and regenerate.
- The list of experiments, their research question and run status is in `experiments/matrix.csv`.
- Relative paths are resolved from the repository root. Seeds are passed at run time (`--seed 42/43/44`).
- `python -m dl_project check --config <file>` validates the structure of a config; it does not validate the
  split or the experiment design.
