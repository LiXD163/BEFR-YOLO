# Contribution Guidelines

BE-FR YOLO is primarily a research reproducibility repository. Contributions
should make experiments easier to inspect, rerun, or audit without changing the
reported manuscript results unless a new result is explicitly documented.

## Issues

- Use issues to report broken commands, missing reproduction metadata, data
  preparation problems, or implementation bugs.
- Include the operating system, Python version, dependency versions, dataset
  YAML, command, and the complete error message.
- Do not attach restricted third-party images, labels, checkpoints, or archives.

## Pull Requests

- Keep pull requests narrowly scoped.
- Explain the motivation, affected files, and expected reproduction impact.
- Add or update tests for new scripts, parsers, or integration behavior.
- Do not overwrite existing result CSV, JSON, Excel, or manuscript-derived
  values unless the PR is explicitly about correcting an extraction error.
- Do not submit third-party datasets, pretrained weights, or files whose license
  is unclear.

## Code Style

- Follow the existing Python style: type hints where helpful, argparse for CLI
  scripts, UTF-8 text output, deterministic sorting for manifests, and clear
  non-zero exits for serious data errors.
- Prefer small utilities that do not mutate original datasets.
- Keep BE-Net, FR-Net, loss, model integration, and training hyperparameters
  stable unless the change is explicitly reviewed as a method change.

## Tests

- Run `python -m compileall .` after code changes.
- Run `pytest` when PyTorch and Ultralytics are installed.
- For data tooling, include a small synthetic fixture or dry-run command that can
  be executed without private datasets.
