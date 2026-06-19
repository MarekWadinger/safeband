"""Streaming classification of sensor fault types.

Extends the root-cause isolation of
``safeband.anomaly.ConditionalGaussianScorer`` — "which signal is
anomalous and in which direction" — to classifying the *type* of
sensor fault from the taxonomy bias / drift / loss of accuracy /
freezing (IDEAS I7).

The taxonomy itself is not new: bias / accuracy_loss (NOISE) / freezing
(CONSTANT/stuck-at), extended with drift, is the canonical
instrumentation-fault taxonomy of Sharma, Golubchik & Govindan (2010,
ACM TOSN, doi:10.1145/1754414.1754419) and Ni et al. (2009). The
contribution here is not the taxonomy but its *streaming, O(1)-memory*
realization on the **conditional** residuals of a self-supervised
detector — typing the fault on the same signal the detector already
uses for localization, with no history buffers, so detection latency
and per-signal attribution are the levers, not the label set.

The classifier consumes, per step, the raw observation together with
the per-signal conditional residuals exposed by
``ConditionalGaussianScorer.residuals_one`` and maintains
exponentially weighted statistics per signal — O(1) memory, no
history buffers:

* **freezing** — stuck-at test on the raw innovation, flagged by
  either of two complementary criteria. A *strict run* requires the
  innovation to stay within ``eps`` (relative to the running signal
  std) over ``freeze_window`` steps — exact for a perfectly stuck
  value. A *variance-collapse* test additionally flags freezing when
  the short-window variance of the innovation collapses to at most
  ``freeze_var_ratio`` times the long-window baseline innovation
  variance *and* the short-window innovation RMS drops below an
  absolute floor (``freeze_abs_scale`` times the stuck-at tolerance);
  this catches a stuck sensor still emitting ~1 LSB readout dither
  (which resets the strict run on every jitter) while the absolute
  floor keeps a bursty-but-healthy real signal that merely goes quiet
  for a window from false-firing. Flagged regardless of the
  conditional score: a frozen value near the conditional mean scores
  ~0.5 forever, so it is invisible to the scorer (the
  ``cond_std -> 0`` blind spot).
* **bias** — persistent constant-sign offset of the normalized
  conditional residual: the short-window mean exceeds
  ``mean_threshold`` while the short-vs-long mean gap stays small.
* **drift** — the short-window residual mean exceeds
  ``mean_threshold`` *and* the signed gap between the short- and
  long-window EWMA means exceeds ``trend_threshold``. This gap is a
  *lagged-mean difference*, not a fitted slope: a still-growing
  deviation keeps the short mean ahead of the slower long mean, whereas
  a settled constant offset lets the long mean catch up and the gap
  closes. The bias/drift boundary therefore depends on the
  ``window``/``long_window`` ratio (the EWMA lag), which the sweep in
  ``examples/07_fault_diagnosis_validation.py`` characterises.
* **accuracy_loss** — variance shift: the short-window residual
  variance exceeds ``var_ratio`` times the long-window baseline
  variance while the mean offset stays near zero.

Single-sensor drift versus system regime change
-----------------------------------------------
A drifting *sensor* diverges from its peers, so its conditional
residual (its value minus what the remaining signals predict for it)
grows without bound. A system *regime change* moves all signals
together, so each conditional mean follows its peers and the
conditional residuals stay small — the scorer adapts instead of
alarming. Two mechanisms encode this distinction:

1. The statistics are computed on *conditional* residuals, which a
   coordinated shift largely cancels.
2. When the scorer's changepoint test fires (its public
   ``drift_detected`` flag is passed in), residual-based detection is
   *attenuated* rather than hard-zeroed for that step: the per-signal
   mean and variance thresholds are scaled up by
   ``suppress_threshold_scale`` while the model re-adapts. A sensor
   moving *with* the regime has a small conditional residual (the
   scorer follows its peers) and stays ``normal`` either way; a real
   sensor fault that co-occurs with a regime change produces a large
   conditional residual that survives the raised threshold and is
   still reported, so a genuine fault is no longer fully masked by an
   unlucky changepoint. Setting ``suppress_threshold_scale`` to
   ``math.inf`` recovers the legacy hard-suppression behaviour.
   Freezing is never suppressed — it is computed on raw innovations.

This resolves the adapt-into-fault tension of the ESwA 2023 paper:
adaptation handles regime changes while the residual-trend test still
exposes a single drifting sensor.

Precedence (documented contract): ``freezing`` wins over all
residual-based labels; ``drift`` wins over ``bias`` via the trend
test; ``bias``/``drift`` (mean offset above threshold) exclude
``accuracy_loss`` (mean offset below threshold) by construction.
"""

import math
from typing import Literal

FaultLabel = Literal[
    "normal",
    "freezing",
    "bias",
    "drift",
    "accuracy_loss",
]

_EPS_FLOOR = 1e-12


class _EwStats:
    """Bias-corrected exponentially weighted mean and variance.

    Tracks first and second moments with decay ``1 - alpha`` and
    divides by the accumulated weight, so early estimates are not
    biased toward the zero initialization.
    """

    def __init__(self, alpha: float) -> None:
        """Initialize empty moment accumulators with decay ``alpha``."""
        self._alpha = alpha
        self._m1 = 0.0
        self._m2 = 0.0
        self._w = 0.0

    def update(self, value: float) -> None:
        """Fold a new value into the weighted moments."""
        a = self._alpha
        self._m1 = (1.0 - a) * self._m1 + a * value
        self._m2 = (1.0 - a) * self._m2 + a * value * value
        self._w = (1.0 - a) * self._w + a

    @property
    def mean(self) -> float:
        """Bias-corrected weighted mean; 0.0 before any update."""
        if self._w == 0.0:
            return 0.0
        return self._m1 / self._w

    @property
    def var(self) -> float:
        """Bias-corrected weighted variance; 0.0 before any update."""
        if self._w == 0.0:
            return 0.0
        mean = self._m1 / self._w
        return max(self._m2 / self._w - mean * mean, 0.0)


class _SignalState:
    """Per-signal streaming state behind the fault tests."""

    def __init__(self, alpha_short: float, alpha_long: float) -> None:
        """Initialize freeze tracking and residual statistics."""
        self.prev: float | None = None
        self.freeze_run = 0
        self.n = 0
        # Number of innovations folded into the innovation baselines.
        self.n_innov = 0
        # Running scale of the raw signal, for the relative freeze eps.
        self.x_stats = _EwStats(alpha_long)
        # Short/long-window statistics of the raw innovation
        # ``x_t - x_{t-1}`` for the variance-collapse freeze test.
        self.innov_short = _EwStats(alpha_short)
        self.innov_long = _EwStats(alpha_long)
        # Short/long-window statistics of the normalized residual.
        self.short = _EwStats(alpha_short)
        self.long_mean = _EwStats(alpha_long)
        self.long_var = _EwStats(alpha_long)


class SensorFaultClassifier:
    """Classify per-signal sensor fault types from streaming residuals.

    Composable with ``ConditionalGaussianScorer``: feed each
    observation together with ``scorer.residuals_one(x)`` and the
    scorer's ``drift_detected`` flag::

        labels = clf.process_one(
            x, scorer.residuals_one(x), scorer.drift_detected
        )

    Residuals are normalized by their conditional std, so all
    mean/variance thresholds are in conditional-sigma units.

    Args:
        window: Effective length of the short statistics window (also
            the warm-up before residual-based labels and the default
            ``freeze_window``).
        long_window: Effective length of the slow baseline window.
            Defaults to ``4 * window``.
        freeze_window: Consecutive near-zero innovations required to
            flag freezing via the strict stuck-at run, and the effective
            length of the short innovation-variance window used by the
            variance-collapse test. Defaults to ``window``.
        freeze_eps: Innovation tolerance relative to the running
            signal std for the strict stuck-at run.
        freeze_var_ratio: Variance-collapse threshold. Freezing is
            flagged when the short-window variance of the raw
            innovation collapses to at most ``freeze_var_ratio`` times
            the long-window baseline innovation variance, sustained
            over the short window. This catches a stuck sensor that
            still emits ~1 LSB readout dither (which resets the strict
            run on every jitter) and, because it compares against the
            signal's own established innovation baseline rather than an
            absolute tolerance, it does not false-fire on a slow but
            healthy signal whose innovations are merely small.
        freeze_abs_scale: Absolute floor for the variance-collapse
            test, as a multiple of the stuck-at tolerance ``eps``. The
            short-window innovation RMS must fall below
            ``freeze_abs_scale * eps`` in addition to clearing the
            ratio gate. This guards against a *bursty but healthy* real
            signal that merely goes quiet for one window — its
            innovation RMS, while small relative to the bursty
            baseline, stays well above the stuck-at floor — so only a
            genuinely near-stuck signal trips the test.
        mean_threshold: Short-window |mean| of the normalized residual
            above which a mean-offset fault (bias or drift) is
            flagged.
        trend_threshold: Signed gap between the short- and long-window
            EWMA residual means (a lagged-mean difference, not a fitted
            slope) above which the offset is classified as drift rather
            than bias. The discriminating boundary scales with the
            ``window``/``long_window`` lag.
        var_ratio: Short/long residual-variance ratio above which a
            near-zero-mean variance shift is flagged as accuracy loss.
        exclusive_attribution: Attribute a mean-offset fault only to
            the signal with the largest |short mean| (and a variance
            fault only to the largest variance ratio) when several
            signals exceed the threshold simultaneously — a fault in
            one sensor also shifts the conditional residuals of its
            peers.
        suppress_on_drift: Attenuate residual-based detection on steps
            where ``drift_detected`` is passed as True (regime change)
            by raising the mean and variance thresholds by
            ``suppress_threshold_scale``, instead of reporting the
            labels at face value. Freezing is never suppressed.
        suppress_threshold_scale: Factor by which ``mean_threshold``
            and ``var_ratio`` are multiplied on changepoint steps when
            ``suppress_on_drift``. The default (5.0) lets a strong
            co-occurring sensor fault survive while masking the
            transient residuals of a coordinated shift. ``math.inf``
            recovers the legacy hard-zeroing of every label.

    Examples:
    --------
    A stuck sensor is flagged regardless of its residual score —
    even a frozen value sitting exactly on the conditional mean
    >>> clf = SensorFaultClassifier(window=4)
    >>> for _ in range(6):
    ...     labels = clf.process_one({"s": 1.0}, {"s": (0.0, 1.0)})
    >>> labels["s"]
    'freezing'

    A persistent constant offset of the conditional residual is bias
    >>> clf = SensorFaultClassifier(window=4, long_window=8)
    >>> x = 0.0
    >>> for _ in range(40):
    ...     x = 1.0 - x
    ...     labels = clf.process_one({"s": x}, {"s": (5.0, 1.0)})
    >>> labels["s"]
    'bias'

    A residual that keeps growing away from the baseline is drift
    >>> clf = SensorFaultClassifier(window=4, long_window=16)
    >>> x = 0.0
    >>> for t in range(60):
    ...     x = 1.0 - x
    ...     labels = clf.process_one({"s": x}, {"s": (0.3 * t, 1.0)})
    >>> labels["s"]
    'drift'

    A variance burst with near-zero mean offset is loss of accuracy
    >>> clf = SensorFaultClassifier(window=4, long_window=16)
    >>> x = 0.0
    >>> for t in range(40):
    ...     x = 1.0 - x
    ...     r = 0.5 if t % 2 else -0.5
    ...     labels = clf.process_one({"s": x}, {"s": (r, 1.0)})
    >>> labels["s"]
    'normal'
    >>> for t in range(8):
    ...     x = 1.0 - x
    ...     r = 4.0 if t % 2 else -4.0
    ...     labels = clf.process_one({"s": x}, {"s": (r, 1.0)})
    >>> labels["s"]
    'accuracy_loss'

    When the scorer reports a regime change, residual-based labels
    are suppressed — coordinated shifts are adaptation, not a fault
    >>> clf = SensorFaultClassifier(window=2, long_window=4)
    >>> x = 0.0
    >>> for _ in range(10):
    ...     x = 1.0 - x
    ...     labels = clf.process_one(
    ...         {"s": x}, {"s": (6.0, 1.0)}, drift_detected=True)
    >>> labels["s"]
    'normal'
    """

    def __init__(
        self,
        window: int = 25,
        long_window: int | None = None,
        freeze_window: int | None = None,
        freeze_eps: float = 1e-3,
        freeze_var_ratio: float = 1e-2,
        freeze_abs_scale: float = 20.0,
        mean_threshold: float = 3.0,
        trend_threshold: float = 1.0,
        var_ratio: float = 4.0,
        exclusive_attribution: bool = True,
        suppress_on_drift: bool = True,
        suppress_threshold_scale: float = 5.0,
    ) -> None:
        """Initialize the classifier and validate the window sizes."""
        if window < 1:
            msg = f"window must be >= 1; got {window}"
            raise ValueError(msg)
        if long_window is None:
            long_window = 4 * window
        if long_window <= window:
            msg = (
                "long_window must exceed window; "
                f"got {long_window} <= {window}"
            )
            raise ValueError(msg)
        self.window = window
        self.long_window = long_window
        self.freeze_window = window if freeze_window is None else freeze_window
        self.freeze_eps = freeze_eps
        self.freeze_var_ratio = freeze_var_ratio
        self.freeze_abs_scale = freeze_abs_scale
        self.mean_threshold = mean_threshold
        self.trend_threshold = trend_threshold
        self.var_ratio = var_ratio
        self.exclusive_attribution = exclusive_attribution
        self.suppress_on_drift = suppress_on_drift
        if suppress_threshold_scale < 1.0:
            msg = (
                "suppress_threshold_scale must be >= 1.0; "
                f"got {suppress_threshold_scale}"
            )
            raise ValueError(msg)
        self.suppress_threshold_scale = suppress_threshold_scale
        self._alpha_short = 2.0 / (window + 1.0)
        self._alpha_long = 2.0 / (long_window + 1.0)
        self._states: dict[str, _SignalState] = {}
        self.labels_: dict[str, FaultLabel] = {}

    def process_one(
        self,
        x: dict[str, float],
        residuals: dict[str, tuple[float, float]] | None = None,
        drift_detected: bool = False,
    ) -> dict[str, FaultLabel]:
        """Update per-signal statistics and return fault labels.

        Args:
            x: Observation keyed by feature name (raw values; used by
                the stuck-at test).
            residuals: Per-signal ``(x_i - cond_mean_i, cond_std_i)``
                as returned by
                ``ConditionalGaussianScorer.residuals_one``. Signals
                with missing, non-finite, or zero-std residuals only
                run the freezing test on that step.
            drift_detected: The scorer's changepoint flag; when True
                (and ``suppress_on_drift``), residual-based labels are
                suppressed for this step.

        Returns:
            Mapping feature name -> fault label from ``{"normal",
            "freezing", "bias", "drift", "accuracy_loss"}``.

        Examples:
        --------
        When two signals exceed the mean threshold together (a fault
        in one sensor also shifts its peers' conditional residuals),
        only the strongest is attributed
        >>> clf = SensorFaultClassifier(window=2, long_window=4)
        >>> x = 0.0
        >>> for _ in range(10):
        ...     x = 1.0 - x
        ...     labels = clf.process_one(
        ...         {"a": x, "b": 1.0 - x},
        ...         {"a": (8.0, 1.0), "b": (4.0, 1.0)})
        >>> labels == {"a": "bias", "b": "normal"}
        True
        """
        candidates: dict[str, FaultLabel] = {}
        for name, value in x.items():
            state = self._states.get(name)
            if state is None:
                state = _SignalState(self._alpha_short, self._alpha_long)
                self._states[name] = state
            frozen = self._update_freeze(state, float(value))
            residual = residuals.get(name) if residuals else None
            if frozen:
                # Skip folding the frozen constant residual into the
                # short/long EWMAs: like x_stats, the residual baselines
                # must reflect only the healthy regime so bias/drift/
                # accuracy tests are not biased on recovery.
                candidates[name] = "freezing"
            else:
                # On a changepoint, attenuate (raise the thresholds)
                # rather than hard-zero, so a strong co-occurring sensor
                # fault is still reported while a coordinated shift is
                # masked.
                scale = (
                    self.suppress_threshold_scale
                    if (drift_detected and self.suppress_on_drift)
                    else 1.0
                )
                candidates[name] = self._update_residual(
                    state,
                    residual,
                    scale,
                )
        if self.exclusive_attribution:
            self._attribute_exclusively(candidates)
        self.labels_ = candidates
        return candidates

    @property
    def diagnostics(self) -> dict[str, dict[str, float]]:
        """Per-signal streaming statistics behind the labels.

        Returns, per signal: number of residual updates ``n``, the
        current ``freeze_run`` length, short/long residual means and
        variances, the ``trend_gap`` (short minus long mean), the
        short/long ``var_ratio`` (NaN until the baseline variance is
        positive) and the ``innov_var_ratio`` (short/long innovation
        variance ratio driving the variance-collapse freeze test; NaN
        until the innovation baseline is positive). The magnitudes
        double as severity measures —
        ``short_mean`` for bias/drift, ``var_ratio`` for accuracy
        loss, ``freeze_run`` for freezing.
        """
        out: dict[str, dict[str, float]] = {}
        for name, state in self._states.items():
            baseline = state.long_var.var
            short_var = state.short.var
            innov_baseline = state.innov_long.var
            out[name] = {
                "n": float(state.n),
                "freeze_run": float(state.freeze_run),
                "short_mean": state.short.mean,
                "short_var": short_var,
                "long_mean": state.long_mean.mean,
                "long_var": baseline,
                "trend_gap": state.short.mean - state.long_mean.mean,
                "var_ratio": (
                    short_var / baseline if baseline > 0.0 else math.nan
                ),
                "innov_var_ratio": (
                    state.innov_short.var / innov_baseline
                    if innov_baseline > _EPS_FLOOR
                    else math.nan
                ),
            }
        return out

    def _update_freeze(self, state: _SignalState, value: float) -> bool:
        """Track raw innovations and return whether the signal froze.

        Two complementary stuck-at tests run on the raw innovation
        ``d = x_t - x_{t-1}``; either one flags freezing:

        * **strict run** — ``|d| <= eps`` (``eps`` relative to the
          running signal std) sustained over ``freeze_window`` steps.
          Exact for a perfectly stuck value.
        * **variance collapse** — the short-window variance of ``d``
          collapses to at most ``freeze_var_ratio`` times the
          long-window baseline variance of ``d`` *and* the short-window
          innovation RMS falls below an absolute floor tied to the
          stuck-at tolerance (``freeze_abs_scale`` times ``eps``). The
          relative test catches a stuck sensor that still emits ~1 LSB
          readout dither (which resets the strict run on every jitter);
          the absolute floor is what keeps the test from false-firing
          on a *bursty but healthy* real signal that merely goes quiet
          for a window — a quiet stretch is far above the stuck-at
          floor even when its ratio to the bursty baseline looks small.
        """
        prev = state.prev
        state.prev = value
        if prev is None:
            state.x_stats.update(value)
            return False
        innov = value - prev
        eps = max(
            self.freeze_eps * math.sqrt(state.x_stats.var),
            _EPS_FLOOR,
        )
        if abs(innov) <= eps:
            state.freeze_run += 1
        else:
            state.freeze_run = 0
        strict_frozen = state.freeze_run >= self.freeze_window

        # Fold the innovation into the short window first, then test it
        # against the long baseline that excludes the current point, so
        # a collapsing run is detected without first contaminating the
        # baseline it is compared against. The collapse must clear both
        # a relative gate (vs the signal's own innovation baseline) and
        # an absolute floor (vs the stuck-at tolerance), so a healthy
        # signal that merely quiets down does not trip it.
        state.innov_short.update(innov)
        baseline = state.innov_long.var
        short_innov_rms = math.sqrt(state.innov_short.var)
        abs_floor = self.freeze_abs_scale * eps
        collapsed = (
            state.n_innov >= self.freeze_window
            and baseline > _EPS_FLOOR
            and state.innov_short.var <= self.freeze_var_ratio * baseline
            and short_innov_rms <= abs_floor
        )
        state.n_innov += 1

        frozen = strict_frozen or collapsed
        if not frozen:
            # Gate the scale and innovation baselines while frozen so a
            # long outage does not erode the tolerance for the healthy
            # regime.
            state.x_stats.update(value)
            state.innov_long.update(innov)
        return frozen

    def _update_residual(
        self,
        state: _SignalState,
        residual: tuple[float, float] | None,
        threshold_scale: float = 1.0,
    ) -> FaultLabel:
        """Fold one conditional residual in and return the candidate.

        ``threshold_scale`` multiplies the mean and variance thresholds
        for this step only — used to attenuate detection during a
        regime change without discarding the residual evidence.
        """
        if residual is None:
            return "normal"
        value, cond_std = residual
        if (
            not math.isfinite(value)
            or not math.isfinite(cond_std)
            or cond_std <= 0.0
        ):
            return "normal"
        z = value / cond_std
        state.n += 1
        state.short.update(z)
        candidate = self._candidate(state, threshold_scale)
        state.long_mean.update(z)
        if candidate != "accuracy_loss":
            # Gate the variance baseline during a variance fault so
            # the inflated noise does not become the new normal.
            state.long_var.update(z)
        return candidate

    def _candidate(
        self,
        state: _SignalState,
        threshold_scale: float = 1.0,
    ) -> FaultLabel:
        """Classify the current residual statistics of one signal.

        ``threshold_scale`` raises the mean and variance thresholds
        (used to attenuate detection during a regime change); the
        trend threshold that separates drift from bias is unscaled.
        """
        if state.n < self.window:
            return "normal"
        mean_threshold = self.mean_threshold * threshold_scale
        var_ratio = self.var_ratio * threshold_scale
        short_mean = state.short.mean
        if abs(short_mean) >= mean_threshold:
            gap = short_mean - state.long_mean.mean
            signed_gap = gap if short_mean >= 0.0 else -gap
            if signed_gap >= self.trend_threshold:
                return "drift"
            return "bias"
        baseline = state.long_var.var
        if baseline > 0.0 and state.short.var >= var_ratio * baseline:
            return "accuracy_loss"
        return "normal"

    def _attribute_exclusively(
        self,
        candidates: dict[str, FaultLabel],
    ) -> None:
        """Keep each fault family on its strongest signal only."""
        mean_faults = [
            name
            for name, label in candidates.items()
            if label in ("bias", "drift")
        ]
        if len(mean_faults) > 1:
            keep = max(
                mean_faults,
                key=lambda name: abs(self._states[name].short.mean),
            )
            for name in mean_faults:
                if name != keep:
                    candidates[name] = "normal"
        var_faults = [
            name
            for name, label in candidates.items()
            if label == "accuracy_loss"
        ]
        if len(var_faults) > 1:

            def ratio(name: str) -> float:
                state = self._states[name]
                baseline = state.long_var.var
                if baseline <= 0.0:
                    return math.inf
                return state.short.var / baseline

            keep = max(var_faults, key=ratio)
            for name in var_faults:
                if name != keep:
                    candidates[name] = "normal"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
