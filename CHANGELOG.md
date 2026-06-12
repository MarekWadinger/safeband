# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- doc-checks pre-commit hooks verifying markdown docs against reality
  (stale make targets, dead paths, unresolvable imports, mermaid syntax)
- Commitizen versioning (`cz bump`) anchored to the publication tags
- Coverage reporting in GitHub Actions (job summary, artifact, Codecov)
- GitHub issues [#53–#63](https://github.com/MarekWadinger/adaptive-interpretable-ad/issues)
  tracking all in-code TODOs, linked from the comments
- `make verify-notebooks`: non-mutating notebook execution gate for
  pre-push (runs to a temp dir instead of rewriting the worktree)
- `functions` and `tests` are real packages (`__init__.py`), fixing
  doctest collection

### Changed

- Packaging migrated from requirements.txt to uv (pyproject + uv.lock)
  [2025-09-02]
- Dropped Python 3.9 support [2025-02-26]; project now requires
  Python >= 3.12
- Lint baseline: ruff `select = ALL` with explicitly permitted ignores
  only — ~1,400 findings resolved (logging instead of prints, pathlib,
  timezone-aware datetimes, numpy `Generator`, full type annotations
  and docstrings)
- CI rebuilt on uv: ruff + ty + pytest on Ubuntu/macOS, Python
  3.12/3.13; the genbadge auto-PR loop is gone
- pre-commit hooks mapped to explicit stages (hygiene/format/docs at
  commit, commitizen at commit-msg, heavy gates at push)
- Generated reports (coverage/junit HTML, badges) no longer tracked
  in git
- README aligned with the renamed repository, uv workflow, and actual
  CLI flags; installation instructions updated [2025-09-02]
- Dockerfile updated for uv-based builds [2025-09-02]

### Fixed

- Metric logging in evaluation always printing `None`
  (`metric.update()` return value was rebound)
- `query_file` not receiving the receiver argument in consumer.py
  [2025-09-02]
- streamz operators broken by renamed keyword parameters
  (`update(x, who, metadata)` is called by keyword upstream)
- `to_mqtt` doctest pointing at a dead public broker
  (mqtt.eclipseprojects.io → test.mosquitto.org)
- Pre-push notebook gate never passing (bare `jupyter` not on hook
  PATH; in-place execution always modified the worktree)

### Removed

- All commented-out code (89 blocks); enforced by an ERA001 pre-push
  gate
- flake8 (fully replaced by ruff)
- `eval()` for type conversion in pipeline params (replaced by an
  explicit converter map)

### Security

- Bumped cryptography to 43.0.1 [2024-10-09] and removed upper-bound
  constraints [2025-02-26]; currently >= 45.0.7

## [0.3.1] - 2024-02-22

### Changed

- Renamed `_feature_names_in` to river-conventional `feature_names_in_`

### Security

- Bumped cryptography from 41.0.6 to 42.0.4

## [2.0.0] - 2024-01-10

Journal publication release — *Adaptable and Interpretable Framework for
Anomaly Detection in SCADA-based industrial systems*, Expert Systems with
Applications (2024), 123200.
DOI: [10.1016/j.eswa.2024.123200](https://doi.org/10.1016/j.eswa.2024.123200)

### Added

- Root cause analysis (RCA) exposure
- Scalability evaluation and ARIMA comparison, with ARIMA support in
  `progressive_val_predict()`
- Multiclass metric support and latency evaluation
- Diagnostics comparison with DBStream; CATS benchmark dataset
- `partial` functions as predefined-parameter pipeline steps
- Graphical abstract and revised ESwA manuscript (two review rounds)
- Python 3.12 compatibility

### Changed

- Bumped river to 0.21.0 and pandas from 1.5.3 to 2.1.2
- Made config file optional (`config.ini` replaced by tracked
  `example.ini`)
- Refactored evaluation: metric evaluation moved into `evaluate.py`
  (`build_fit_evaluate()`)

### Fixed

- Numeric instability issues
- Pipeline and forecasting-model evaluation in
  `progressive_val_predict()`, including detectors whose `learn_one`
  does not return `self`
- Recipient argument name in encrypted queries

### Security

- Bumped cryptography from 41.0.4 to 41.0.6

## [0.3.0] - 2023-10-19

### Added

- Added DOI to final version of publication
- Added file output to config.ini
- Added ConditionalGaussianScorer [2023-07-26], now the default scorer
- Pipeline support for GaussianScorer
- Model save/load recovery on error
- Docker deployment (Dockerfile, docker-compose, prebuilt river wheels
  for Linux containers)
- Local Outlier Factor detector for comparison studies
- Case-study and benchmark data for reproducibility
- ESwA manuscript materials [2023-09-18]

### Changed

- Update project structure with procedure call convention
  - dynamic_signal_limits_service.py &rarr; server.py; client.py
  - query_signal_limits.py &rarr; consumer.py
- Moved all notebooks to examples
- Made consumer.py use config.ini
- Made encryption optional
- Adapted MQTT handling to the electrometer deployment
- Moved learning into the parent scorer class
- Aligned README.md
- Changed the way dynamic limits are computed (1/3 slower) [2023-07-25]

### Fixed

- Wrong pairing of limit values
- Numeric instability caused by `VAR_SMOOTHING` (removed)
- Change-point adaptation
- `limit_one` on an uninitialized model

### Security

- Bumped cryptography from 41.0.3 to 41.0.4

## [0.2.0] - 2023-06-23

### Added

- multivariate detection support: `MultivariateGaussian` distribution
  contributed for river [2023-03-29]
- `progressive_val_predict()` evaluation function with examples
  [2023-04-03]
- [multivariate_gaussian.ipynb](https://github.com/MarekWadinger/adaptive-interpretable-ad/blob/main/multivariate_gaussian.ipynb) notebook with example of usage.
- encryption of communication (RSA key generation/checking, signing,
  and verification, with test scenarios)

### Changed

- [dynamic_signal_limits_service.py](https://github.com/MarekWadinger/adaptive-interpretable-ad/blob/main/dynamic_signal_limits_service.py) signs and encrypts sent messages
- [query_signal_limits.py](https://github.com/MarekWadinger/adaptive-interpretable-ad/blob/main/query_signal_limits.py) decrypts and verifies received messages
- `GaussianScorer` refactored into its own module [2023-03-29]

## [1.0.0] - 2023-06-21

Conference publication release (tagged retroactively, same snapshot as
0.1.0) — *Real-Time Outlier Detection with Dynamic Process Limits*, 24th
International Conference on Process Control (PC), 2023.
DOI: [10.1109/PC58330.2023.10217717](https://doi.org/10.1109/PC58330.2023.10217717)

## [0.1.0] - 2023-06-21

### Added

- [dynamic_signal_limits_service.py](https://github.com/MarekWadinger/adaptive-interpretable-ad/blob/main/dynamic_signal_limits_service.py)
and
[query_signal_limits.py](https://github.com/MarekWadinger/adaptive-interpretable-ad/blob/main/query_signal_limits.py)
for online detection and dynamic process limits estimation and querying.
- [online_outlier_detection.ipynb](https://github.com/MarekWadinger/adaptive-interpretable-ad/blob/main/online_outlier_detection.ipynb)
notebook with example of usage.
- Doctests and
[pytests](https://github.com/MarekWadinger/adaptive-interpretable-ad/tree/main/tests).
- HTML and badges for Reports on
[code coverage](https://codecov.io/gh/MarekWadinger/adaptive-interpretable-ad),
[tests](https://htmlpreview.github.io/?https://github.com/MarekWadinger/adaptive-interpretable-ad/blob/main/reports/junit/report/index.html),
and
[linting](https://htmlpreview.github.io/?https://github.com/MarekWadinger/adaptive-interpretable-ad/blob/main/reports/flake8/report/index.html).
- [Publication](https://github.com/MarekWadinger/adaptive-interpretable-ad/tree/main/publications)
files for papers and presentations (PC2023, CDC2023).

[unreleased]: https://github.com/MarekWadinger/adaptive-interpretable-ad/compare/0.3.1...HEAD
[0.3.1]: https://github.com/MarekWadinger/adaptive-interpretable-ad/compare/2.0.0...0.3.1
[2.0.0]: https://github.com/MarekWadinger/adaptive-interpretable-ad/compare/0.3.0...2.0.0
[0.3.0]: https://github.com/MarekWadinger/adaptive-interpretable-ad/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/MarekWadinger/adaptive-interpretable-ad/compare/1.0.0...0.2.0
[1.0.0]: https://github.com/MarekWadinger/adaptive-interpretable-ad/releases/tag/1.0.0
[0.1.0]: https://github.com/MarekWadinger/adaptive-interpretable-ad/releases/tag/0.1.0
