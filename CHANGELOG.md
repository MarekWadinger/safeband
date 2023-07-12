# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added DOI to final version of publication
- Added file output to config.ini

### Changed

- Update project structure with procedure call convention
  - dynamic_signal_limits_service.py &rarr; server.py; client.py
  - query_signal_limits.py &rarr; consumer.py
- Moved all notebooks to examples
- Made consumer.py use config.ini
- Aligned README.md

## [0.2.0] - 2023-06-23

### Added

- multivariate detection support
- [multivariate_gaussian.ipynb](https://github.com/MarekWadinger/online_outlier_detection/blob/main/multivariate_gaussian.ipynb) notebook with example of usage.
- encryption of communication

### Changed

- [dynamic_signal_limits_service.py](https://github.com/MarekWadinger/online_outlier_detection/blob/main/dynamic_signal_limits_service.py) signs and encrypts sent messages
- [query_signal_limits.py](https://github.com/MarekWadinger/online_outlier_detection/blob/main/query_signal_limits.py) decrypts and verifies received messages

## [0.1.0] - 2023-06-23

### Added

- [dynamic_signal_limits_service.py](https://github.com/MarekWadinger/online_outlier_detection/blob/main/dynamic_signal_limits_service.py)
and
[query_signal_limits.py](https://github.com/MarekWadinger/online_outlier_detection/blob/main/query_signal_limits.py)
for online detection and dynamic process limits estimation and querying.
- [online_outlier_detection.ipynb](https://github.com/MarekWadinger/online_outlier_detection/blob/main/online_outlier_detection.ipynb)
notebook with example of usage.
- Doctests and
[pytests](https://github.com/MarekWadinger/online_outlier_detection/tree/main/tests).
- HTML and badges for Reports on
[code coverage](https://codecov.io/gh/MarekWadinger/online_outlier_detection),
[tests](https://htmlpreview.github.io/?https://github.com/MarekWadinger/online_outlier_detection/blob/main/reports/junit/report/index.html),
and
[linting](https://htmlpreview.github.io/?https://github.com/MarekWadinger/online_outlier_detection/blob/main/reports/flake8/report/index.html).
- [Publication](https://github.com/MarekWadinger/online_outlier_detection/tree/main/publications)
files for papers and presentations.

[unreleased]: https://github.com/MarekWadinger/online_outlier_detection/compare/0.2.0...HEAD
[0.2.0]: https://github.com/MarekWadinger/online_outlier_detection/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/MarekWadinger/online_outlier_detection/releases/tag/0.1.0releases/tag/0.1.0