## 2.2.2 (2026-06-12)

### Fix

- honor out_topics, reject debug with remote brokers, cover sinks
- make scorer contracts explicit for edge-case inputs
- seed multivariate CDF evaluation for reproducible scores
- confirm MQTT delivery instead of fire-and-forget publishing
- return first valid type from union type hints

### Refactor

- subclass TimeRolling for buffer length instead of monkey-patching

## 2.2.1 (2026-06-12)

### Fix

- **ci**: skip plot-only notebook in gate, pin codecov project target
- scope CI notebook gate to notebooks with committed data
- pin setup-uv to exact v8.2.0 (no floating v8 tag exists)
- wire Pulsar source instead of stale Python version gate
- surface service logs and guard consumer against missing key_path
- package structure (INP001) and repair autofix-broken doctests
- resolve bugbear findings (B007, B023)
- timezone-aware datetime handling (DTZ)

### Refactor

- remove commented-out code (ERA001)
- final lint cleanup (E501, COM812, SIM, S307, PLC, YTT, UP, TRY401)
- add type annotations (ANN rules)
- migrate to numpy random Generator (NPY002, S311)
- misc lint cleanups (A001, A004, TC010, TRY004, PLC0415, ARG, E501, TD002)
- avoid redefining loop variables (PLW2901)
- use to_numpy() instead of .values (PD011)
- migrate filesystem operations to pathlib (PTH)
- replace removed print output with logging (T201)
- apply ruff --select ALL --unsafe-fixes autofixes and ty 0.0.48 type fixes

## 2.2.0 (2025-09-02)

### Fix

- pass receiver argument to query_file function in consumer.py
- old lock

### Refactor

- standardize docstring formatting across multiple files

### Build

- utilize astral bundle and move deps to pyproject + uv.lock
- update Dockerfile and dependencies for improved builds
- replace utcnow with now

## 2.1.0 (2025-02-26)

### Fix

- ensure runnability of examples

### Refactor

- typing and formatting
- freshen up the dev-related files

### Build

- drop Python 3.9 support
- bump requirements
- remove cryptography upper-bound constraints

### Docs

- update docker install instructions

## 2.0.1 (2024-02-22)

### Refactor

- rename _feature_names_in to river-conventional feature_names_in_

### Build

- bump cryptography from 41.0.6 to 42.0.4

## 2.0.0 (2024-01-10)

Journal publication release — *Adaptable and Interpretable Framework for
Anomaly Detection in SCADA-based industrial systems*, Expert Systems with
Applications (2024), 123200.
DOI: [10.1016/j.eswa.2024.123200](https://doi.org/10.1016/j.eswa.2024.123200)

### Feat

- expose root cause analysis (RCA)
- scalability evaluation and ARIMA comparison, with ARIMA support in
  progressive_val_predict()
- multiclass metric support and latency evaluation
- diagnostics comparison with DBStream and CATS benchmark dataset
- partial functions as predefined-parameter pipeline steps
- Python 3.12 compatibility

### Fix

- numeric instability issues
- pipeline and forecasting-model evaluation in progressive_val_predict(),
  including detectors whose learn_one does not return self
- recipient argument name in encrypted queries

### Refactor

- move metric evaluation into evaluate.py (build_fit_evaluate())

### Build

- bump river to 0.21.0 and pandas from 1.5.3 to 2.1.2
- make config file optional (config.ini replaced by tracked example.ini)
- bump cryptography from 41.0.4 to 41.0.6

## 1.2.0 (2023-10-19)

### Feat

- ConditionalGaussianScorer, used as the default scorer
- Pipeline support for GaussianScorer
- save and load model on error (recovery)
- file output configurable via config.ini
- Local Outlier Factor detector for comparison studies
- case-study and benchmark data for reproducibility
- optional encryption
- Docker deployment (Dockerfile, docker-compose, river wheels for
  Linux containers)
- ESwA manuscript materials

### Fix

- wrong pairing of limit values
- numeric instability caused by VAR_SMOOTHING (removed)
- change-point adaptation
- limit_one on an uninitialized model

### Refactor

- restructure to procedure call convention:
  dynamic_signal_limits_service.py to server.py and client.py,
  query_signal_limits.py to consumer.py
- move all notebooks to examples
- move learning into the parent scorer class
- change the way dynamic limits are computed (1/3 slower)

### Build

- bump cryptography from 41.0.3 to 41.0.4

## 1.1.0 (2023-06-23)

### Feat

- multivariate detection support (MultivariateGaussian distribution
  contributed for river)
- progressive_val_predict() evaluation function with examples
- encrypted communication (RSA key generation and checking, signing,
  and verification) with test scenarios

### Refactor

- GaussianScorer into its own module

## 1.0.0 (2023-06-21)

Conference publication release — *Real-Time Outlier Detection with
Dynamic Process Limits*, 24th International Conference on Process
Control (PC), 2023.
DOI: [10.1109/PC58330.2023.10217717](https://doi.org/10.1109/PC58330.2023.10217717)

### Feat

- dynamic_signal_limits_service.py and query_signal_limits.py for
  online detection, dynamic process limits estimation, and querying
- example notebook for online outlier detection
- doctests and pytests
- reports and badges for coverage, tests, and linting
- publication materials (PC2023, CDC2023)
