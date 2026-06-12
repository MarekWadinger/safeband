# Research Ideas → Code & Paper Mapping

Mapping of 7 implementation ideas to the codebase and the ESwA paper
([publications/ESwA2023/](publications/ESwA2023/), sections in
[publications/ESwA2023/sections/](publications/ESwA2023/sections/)).

## Priority table

Sorted by **value first, effort second** (best value-for-effort on top).

| Code | Idea | Type | Value | Effort | Rationale |
|------|------|------|-------|--------|-----------|
| **I3** | Fix encryption returns limits as list/str | fix | High | Low | Correctness bug in the deployed pipeline; consumers receive stringified limits they cannot use numerically |
| **I4** | Option to provide physical limits | feat | High | Medium | Paper explicitly promises combination with static SCADA limits; a `TODO` for it already sits in the code |
| **I2** | Make part of `GaussianScorer` public | refactor | Medium | Low | Per-signal scores and changepoint signal are the paper's headline diagnostics, yet are private API |
| **I1** | Protect anomaly detector with ThresholdFilter | refactor | Medium | Medium | Cleaner river-idiomatic design of the self-supervised protection; no new capability |
| **I6** | TSB-AD benchmark comparison | benchmark | High | High | Modernizes the SKAB-only validation; strong material for thesis / follow-up paper |
| **I7** | Sensor fault diagnosis taxonomy (bias, drift, accuracy loss, freezing) | feat / research | High | Very High | Genuine research extension; AID currently misses freezing and can *adapt into* drift faults |
| **I5** | Compare with Reunanen et al. 2020 (s41060-019-00191-3) | benchmark | Medium | High | One extra baseline; likely needs reimplementation from the paper |

---

## I1 — Refactor: protect anomaly detector with ThresholdFilter

**What:** Replace the hand-rolled "learn only on normal samples unless changepoint"
logic inside `GaussianScorer` with a composable wrapper in the spirit of
`river.anomaly.ThresholdFilter` / `QuantileFilter` (subclass of
`river.anomaly.base.AnomalyFilter`), so protection becomes a detector-agnostic
decorator instead of a constructor flag.

**Code today:**
- `protect_anomaly_detector` flag: [functions/anomaly.py:215](functions/anomaly.py#L215) and [functions/anomaly.py:269-275](functions/anomaly.py#L269-L275) (buffer construction: `collections.deque` or `TimeRolling(Store())`)
- Protection gate in `learn_one`: [functions/anomaly.py:321-330](functions/anomaly.py#L321-L330) — `predict_one` → buffer → `_drift_detected()` → conditional `_learn_one`
- Changepoint test `_drift_detected`: [functions/anomaly.py:296-301](functions/anomaly.py#L296-L301)
- Helper `Store` class used only for the time-based buffer: [functions/anomaly.py:45-94](functions/anomaly.py#L45-L94)
- Duplicated gate in `process_one` (`if not is_anomaly: learn_one`): [functions/anomaly.py:445-449](functions/anomaly.py#L445-L449) — note this **double-protects** when `protect_anomaly_detector=True`, a latent inconsistency the refactor should resolve
- Same flag threaded through `ConditionalGaussianScorer`: [functions/anomaly.py:534](functions/anomaly.py#L534), [functions/anomaly.py:547-553](functions/anomaly.py#L547-L553)

**Paper:**
- Self-supervised training and the two protection mechanisms: [sections/proposed_method.tex:36-44](publications/ESwA2023/sections/proposed_method.tex#L36-L44)
- Changepoint test (Eq. `eq:changepoint`) and adaptation period $t_a$: [sections/proposed_method.tex:23-29](publications/ESwA2023/sections/proposed_method.tex#L23-L29)
- $t_a = 1/4\,t_e$ tuning guidance: [sections/proposed_method.tex:55](publications/ESwA2023/sections/proposed_method.tex#L55)

**Notes:** river's stock `ThresholdFilter.learn_one` only skips learning on anomalous
samples — it has **no changepoint re-adaptation**. The clean target is a custom
`AdaptiveThresholdFilter(AnomalyFilter)` that owns the buffer + Eq. (changepoint)
logic, leaving `GaussianScorer` pure. Keeps the paper's algorithm intact, improves
composability with any river detector.

---

## I2 — Refactor: make part of `GaussianScorer` public

**What:** Promote private methods that carry the paper's interpretability claims to
public, documented API (and a step toward upstreaming to `river`).

**Code today (private candidates):**
- `_scores_one` — per-signal conditional CDF scores, the core of root-cause isolation: [functions/anomaly.py:578-596](functions/anomaly.py#L578-L596)
- `_score_one` — score + index of farthest-from-center signal: [functions/anomaly.py:598-610](functions/anomaly.py#L598-L610)
- `_drift_detected` — changepoint signal, valuable as a user-facing "regime change" indicator: [functions/anomaly.py:296-301](functions/anomaly.py#L296-L301)
- `_get_limits` — per-signal limit computation from conditional moments: [functions/anomaly.py:635-641](functions/anomaly.py#L635-L641)
- `_farthest_from_center` — root-cause ranking helper: [functions/anomaly.py:558-576](functions/anomaly.py#L558-L576)
- Already-public diagnostics for contrast: `get_root_cause` [functions/anomaly.py:612-613](functions/anomaly.py#L612-L613), `limit_one` [functions/anomaly.py:358-421](functions/anomaly.py#L358-L421), `process_one` [functions/anomaly.py:423-451](functions/anomaly.py#L423-L451)
- The conditional-distribution engine `MultivariateGaussian.mv_conditional` is already public: [functions/proba.py:52-115](functions/proba.py#L52-L115)

**Paper:**
- "Diagnostics" §: root-cause isolation via per-signal conditional probabilities — exactly what `_scores_one` computes: [sections/proposed_method.tex:51-52](publications/ESwA2023/sections/proposed_method.tex#L51-L52)
- The three diagnostic mechanisms (outlier per signal, drift, dynamic limits): [sections/proposed_method.tex:3-5](publications/ESwA2023/sections/proposed_method.tex#L3-L5)

**Notes:** a public `scores_one(x) -> dict[str, float]` (keyed by feature name instead
of positional list) would directly expose the paper's diagnostics and make ranked
root causes (top-k, not just argmax) possible. Low risk: additive API change.

---

## I3 — Fix: encryption returns limits as list/str

**What:** `level_high` / `level_low` survive the sign→encrypt→decrypt→verify round
trip only as **strings** (e.g. `"{'a': 0.5, 'b': 0.6}"` or `"[0.5, -0.5]"`), never
parsed back to dict/float, because the crypto layer coerces every non-str value with
`str(v)`.

**Code today (the round trip):**
- Limits produced as float-or-dict: [rpc_server.py:233-245](rpc_server.py#L233-L245) (`fit_transform` returns `level_high`/`level_low` from `model.process_one`)
- Sink pipeline `sign_data → encrypt_data`: [rpc_server.py:524-526](rpc_server.py#L524-L526)
- Lossy coercion on encrypt: `str(v)` for non-str dict values: [functions/encryption.py:164-178](functions/encryption.py#L164-L178) (esp. lines 169-171)
- Same coercion on sign: [functions/encryption.py:275-291](functions/encryption.py#L275-L291) (esp. lines 279-284)
- Chunking of long payloads into **lists** of ciphertext blocks: `split_msg` [functions/encryption.py:88-103](functions/encryption.py#L88-L103), applied at [functions/encryption.py:174-176](functions/encryption.py#L174-L176)
- Decrypt returns flat bytes/str with no type recovery: [functions/encryption.py:192-238](functions/encryption.py#L192-L238), `decode_data` stringifies numbers too: [functions/encryption.py:469-488](functions/encryption.py#L469-L488)
- Consumer receives the still-stringly-typed item: [consumer.py:51-60](consumer.py#L51-L60) and [consumer.py:82-91](consumer.py#L82-L91) (`verify_and_decrypt_data`, [functions/encryption.py:349-378](functions/encryption.py#L349-L378))
- Test fixture documenting current (stringified) expectations: [tests/test_encryption.py](tests/test_encryption.py)

**Paper:** the encrypted RPC pipeline is an implementation artifact of the deployment
story (SCADA/IoT integration, [sections/proposed_method.tex:5](publications/ESwA2023/sections/proposed_method.tex#L5),
case studies in [sections/case_study.tex:1](publications/ESwA2023/sections/case_study.tex#L1));
the paper itself doesn't specify the wire format — the bug is purely code-side.

**Suggested fix:** serialize the payload once with `json.dumps` before
`sign_data`/`encrypt_data` and `json.loads` after `verify_and_decrypt_data`, so type
fidelity is JSON's problem, not `str()`'s. Touches ~4 call sites; add a round-trip
test asserting `isinstance(item["level_high"], (float, dict))`.

---

## I4 — Feat: option to provide physical limits

**What:** Let the user pass known physical/design bounds (sensor range, actuator
saturation) so dynamic limits are clipped to them — and values outside physical
bounds are flagged regardless of the learned distribution.

**Code today:**
- The TODO is already in the code: `# TODO: consider strict process boundaries` at [functions/anomaly.py:373-375](functions/anomaly.py#L373-L375), inside `GaussianScorer.limit_one` ([functions/anomaly.py:358-421](functions/anomaly.py#L358-L421))
- Conditional variant to extend: `ConditionalGaussianScorer.limit_one` [functions/anomaly.py:643-667](functions/anomaly.py#L643-L667) and `_get_limits` [functions/anomaly.py:635-641](functions/anomaly.py#L635-L641)
- Constructor where `physical_limits: dict[str, tuple[float, float]]` would land: [functions/anomaly.py:205-216](functions/anomaly.py#L205-L216) (and [functions/anomaly.py:528-535](functions/anomaly.py#L528-L535))
- Model params parsing for service config: [rpc_server.py:46](rpc_server.py#L46) (`expand_model_params`)

**Paper:**
- "Dynamic limits acquisition" §, which states the limits *"may be used as an addition to static operating limits used by monitoring systems in SCADA"* — this feature closes that gap in code: [sections/proposed_method.tex:46-49](publications/ESwA2023/sections/proposed_method.tex#L46-L49)
- Limit violation = anomaly: end of [sections/proposed_method.tex:49](publications/ESwA2023/sections/proposed_method.tex#L49)

**Notes:** semantics to decide: (a) clip reported `thresh_high/low` into
`[phys_low, phys_high]`, (b) force `predict_one = 1` when `x` violates physical
bounds even during grace period, (c) optionally exclude physically-impossible samples
from learning (synergy with I1's filter).

---

## I5 — Benchmark: compare with Reunanen et al. 2020

*"Unsupervised online detection and prediction of outliers in streams of sensor
data"* ([doi:10.1007/s41060-019-00191-3](https://doi.org/10.1007/s41060-019-00191-3))

**What:** Add the online SVM/LSTM-based outlier detection+prediction pipeline of
Reunanen et al. as a baseline next to OC-SVM and HS-Trees.

**Code today:**
- Existing comparison harness: [examples/comparison.ipynb](examples/comparison.ipynb) (OC-SVM, HS-Trees on SKAB), [examples/comparison_ARIMA.ipynb](examples/comparison_ARIMA.ipynb), [examples/comparison_diagnostics.py](examples/comparison_diagnostics.py)
- Evaluation loop to plug into: `progressive_val_predict` [functions/evaluate.py:18](functions/evaluate.py#L18), metrics writer `save_evaluate_metrics` [functions/evaluate.py:217](functions/evaluate.py#L217), `build_fit_evaluate` [functions/evaluate.py:299](functions/evaluate.py#L299)

**Paper:**
- "Real Data Benchmark" § — where this baseline would slot in: [sections/case_study.tex:95-105](publications/ESwA2023/sections/case_study.tex#L95-L105)
- Results table discussion (F1, precision, recall, FAR, latency): [sections/case_study.tex:136](publications/ESwA2023/sections/case_study.tex#L136)
- **Not yet cited** in [publications/ESwA2023/main.bib](publications/ESwA2023/main.bib) (checked — no Reunanen / s41060 entry); add the reference if pursued

**Notes:** Reunanen et al. is conceptually the closest related work (online,
unsupervised, sensor streams, predicts outliers ahead) — valuable for the thesis
related-work positioning even if no public reference implementation exists (expect
reimplementation effort; hence High effort, Medium value vs. I6).

---

## I6 — Benchmark: TSB-AD

([github.com/TheDatumOrg/TSB-AD](https://github.com/TheDatumOrg/TSB-AD) — VLDB 2024
time-series anomaly detection benchmark, 1000+ labeled series, 40 algorithms)

**What:** Evaluate AID on TSB-AD-U (univariate) / TSB-AD-M (multivariate) for a
modern, large-scale, reproducible comparison beyond SKAB.

**Code today:**
- Same harness as I5: [functions/evaluate.py:18](functions/evaluate.py#L18) (`progressive_val_predict`), [functions/evaluate.py:275](functions/evaluate.py#L275) (`batch_save_evaluate_metrics`)
- Existing benchmark entry points to imitate: [examples/comparison.ipynb](examples/comparison.ipynb), scalability protocol [examples/05_scalability.py](examples/05_scalability.py) + [examples/05_scalability_eval.ipynb](examples/05_scalability_eval.ipynb)
- Models for both regimes already exist: `GaussianScorer` (univariate) [functions/anomaly.py:101](functions/anomaly.py#L101), `ConditionalGaussianScorer` (multivariate) [functions/anomaly.py:454](functions/anomaly.py#L454)

**Paper:**
- Current validation is SKAB-only, motivated by *"no established benchmarking multivariate data were found"*: [sections/case_study.tex:99](publications/ESwA2023/sections/case_study.tex#L99) — TSB-AD is precisely the missing established benchmark (postdates the paper)
- Evaluation protocol to mirror (Bayesian-optimized hyperparams, online preprocessing): [sections/case_study.tex:101-105](publications/ESwA2023/sections/case_study.tex#L101-L105)

**Notes:** TSB-AD ships its own evaluation metrics (VUS-PR/VUS-ROC) — adopting their
protocol verbatim maximizes comparability and citability. Caveat: TSB-AD is largely
batch-oriented; AID's streaming/one-pass constraint becomes a *selling point* if
reported honestly (one pass, no train/test leakage).

---

## I7 — Feat: sensor fault diagnosis taxonomy

(bias, drift, loss of accuracy, freezing — per the four-panel figure of
actual-vs-measured value over time)

**What:** Extend AID's diagnosis from "which signal is anomalous + direction" to
classifying the *type* of sensor fault.

**Capability mapping per fault type:**

| Fault | AID today | Gap / extension |
|-------|-----------|-----------------|
| (a) Bias | Detected as conditional outlier; root cause isolated via `_scores_one` [functions/anomaly.py:578-596](functions/anomaly.py#L578-L596) | Classify as bias when the conditional deviation is a *persistent constant offset* |
| (b) Drift | Changepoint adaptation `_drift_detected` [functions/anomaly.py:296-301](functions/anomaly.py#L296-L301) may **adapt into the fault** — the model accepts the drifting sensor as the new normal | Distinguish system regime change (adapt) from single-sensor drift (alarm); residual-trend test on conditional mean |
| (c) Loss of accuracy | Rolling `sigma` absorbs the inflated variance over the expiration window $t_e$ | Variance-shift detector on per-signal conditional residuals |
| (d) Freezing | **Missed**: a frozen value near the conditional mean stays inside limits indefinitely; `cond_std → 0` branch returns score 0.0 at [functions/anomaly.py:590-595](functions/anomaly.py#L590-L595) | Stuck-at test (zero innovation over window); cheap and high-value first increment |

**Code anchors:**
- Root-cause machinery to build on: `get_root_cause` [functions/anomaly.py:612-613](functions/anomaly.py#L612-L613), `predict_one` root-cause assignment [functions/anomaly.py:619-633](functions/anomaly.py#L619-L633)
- Conditional moments per signal (`mv_conditional`): [functions/proba.py:52-115](functions/proba.py#L52-L115)
- Diagnosis demo to extend: [examples/comparison_diagnostics.py](examples/comparison_diagnostics.py), [examples/03_conditional_ae_2023.ipynb](examples/03_conditional_ae_2023.ipynb)

**Paper:**
- "Diagnostics" § — currently root-cause isolation + deviation direction/extent only; fault-type classification is the natural next claim: [sections/proposed_method.tex:51-52](publications/ESwA2023/sections/proposed_method.tex#L51-L52)
- The three existing mechanisms the taxonomy would become the fourth of: [sections/proposed_method.tex:3-5](publications/ESwA2023/sections/proposed_method.tex#L3-L5)
- Tension to resolve in writing: adaptation speed vs. drift-fault detection, cf. $t_a$ trade-off discussion [sections/proposed_method.tex:55](publications/ESwA2023/sections/proposed_method.tex#L55)

**Notes:** this is follow-up-paper material (fault *identification*, not just
detection). Suggested staging: freezing detector first (trivial, fills a real blind
spot), then bias-vs-drift discrimination on conditional residuals, then accuracy-loss
via variance tracking.

---

## Suggested execution order

1. **I3** (unblocks trustworthy downstream consumption; small diff)
2. **I2** (small API change; makes I7 easier by exposing per-signal scores)
3. **I4** (closes the paper's SCADA promise; uses I2's public limits path)
4. **I1** (refactor while touching `learn_one` anyway; resolves the double-protection inconsistency)
5. **I6** (benchmark with highest publication value)
6. **I7** (research extension building on I2's public diagnostics)
7. **I5** (add only if the related-work positioning needs the head-to-head)
