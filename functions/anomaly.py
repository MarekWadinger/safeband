"""Anomaly detection scorers based on Gaussian distribution models."""

import collections
import warnings
from collections.abc import Iterator, Sized
from datetime import datetime, timedelta
from typing import Protocol, Self, cast, runtime_checkable

import numpy as np
import pandas as pd
from river import anomaly, utils
from river.utils import Rolling, TimeRolling
from scipy.stats import norm


@runtime_checkable
class Distribution(Protocol):  # pragma: no cover
    """Protocol for a univariate or multivariate distribution estimator."""

    mu: float | dict[str, float]
    sigma: float | pd.DataFrame
    n_samples: float | int

    def update(self, x: dict[str, float]) -> None:
        """Update the distribution with a new observation."""
        ...

    def cdf(self, x: dict[str, float]) -> float:
        """Return the cumulative distribution function value."""
        ...


@runtime_checkable
class ConditionableDistribution(
    Distribution,
    Protocol,
):  # pragma: no cover
    """Protocol for a distribution that supports conditional computation."""

    mu: dict[str, float]
    sigma: pd.DataFrame
    var: pd.DataFrame
    n_samples: float | int

    def update(self, x: dict[str, float]) -> None:
        """Update the distribution with a new observation."""
        ...

    def cdf(self, x: dict[str, float]) -> float:
        """Return the cumulative distribution function value."""
        ...

    def mv_conditional(
        self,
        observed_values: dict[str, float] | np.ndarray,
        var_idx: str | int,
        mean: dict[str, float] | np.ndarray,
        covariance: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return conditional mean, covariance, and std for a variable."""
        ...


class Store:
    """A custom store that implements basic list-like functionality.

    This class provides a custom store with list-like behavior. It supports
    iteration, indexing, length calculation, appending, updating, and reverting
    operations.

    Example:
    >>> my_store = Store()
    >>> my_store.append(1)
    >>> my_store.append(2)
    >>> my_store.append(3)
    >>> my_store[1]
    2
    >>> len(my_store)
    3
    >>> for item in my_store:
    ...     print(item)
    1
    2
    3
    >>> my_store.update(4)
    >>> my_store[-1]
    4
    >>> my_store.revert()
    >>> len(my_store)
    3

    """

    def __init__(self) -> None:
        """Initialize an empty store."""
        self.x: list = []

    def __iter__(self) -> Iterator[float]:
        """Yield items in insertion order."""
        yield from self.x

    def __getitem__(self, idx: int) -> float:
        """Return the item at the given index."""
        return self.x[idx]

    def __len__(self) -> int:
        """Return the number of stored items."""
        return len(self.x)

    def append(self, *args: float, **_kwargs: object) -> None:
        """Append the first positional argument to the store."""
        self.x.append(args[0])

    def update(self, *args: float, **_kwargs: object) -> None:
        """Append the first positional argument to the store."""
        self.x.append(args[0])

    def revert(self, *_args: float, **_kwargs: object) -> None:
        """Remove the oldest item from the store."""
        self.x.pop(0)


class TimeRollingBuffer(TimeRolling):
    """TimeRolling window that reports the length of its underlying store.

    ``len`` looks ``__len__`` up on the type, so river's attribute
    delegation to the wrapped object never applies to it; defining it on a
    subclass replaces the former monkey-patch on ``TimeRolling``.
    """

    def __len__(self) -> int:
        """Return the number of items currently held by the store."""
        return len(cast("Store", self.obj))

    def __iter__(self) -> Iterator[float]:
        """Yield the stored items in insertion order."""
        return iter(cast("Store", self.obj))


class GaussianScorer(anomaly.base.AnomalyDetector):
    """Gaussian Scorer for anomaly detection.

    Parameters
    ----------
        threshold (float): Anomaly threshold.
        log_threshold (float): Controls the logarithmic threshold to manage
        small values of lower threshold.
        window_size (int or None): Size of the rolling window.
        period (int or None): Time period for time rolling.
        grace_period (int): Grace period before scoring starts.
        physical_limits (tuple[float, float] or None): Known static
        bounds of the modeled signal. Reported dynamic limits are
        clipped into them, and observations outside them are flagged
        anomalous even during the grace period.
        learn_on_physical_violation (bool): Whether physically
        impossible samples still update the distribution. Defaults to
        False, excluding them from learning entirely.

    Examples:
    --------
    Make sure that the passed distribution sattisfies necessary protocol
    >>> bad_scorer = GaussianScorer(
    ...     type('Dist', (object,), {})(), grace_period=0
    ...     )
    Traceback (most recent call last):
    ...
    TypeError: ... does not satisfy the necessary protocol

    Gaussian scorer on rolling window
    >>> from river.utils import Rolling
    >>> from river.proba import Gaussian
    >>> scorer = GaussianScorer(Rolling(Gaussian(), window_size=3),
    ...     grace_period=2)
    >>> isinstance(scorer, GaussianScorer)
    True
    >>> scorer.gaussian.mu
    0.0
    >>> scorer.score_one(2.4715629565996924)
    0.5
    >>> scorer.limit_one()
    (np.float64(nan), np.float64(nan))
    >>> scorer.learn_one(1).gaussian.mu
    1.0
    >>> scorer.gaussian.sigma
    0.0
    >>> scorer.learn_one(0).gaussian.sigma
    0.7071067811865476
    >>> scorer.limit_one()
    (np.float64(2.4715629565996924), np.float64(-1.4715629565996926))
    >>> scorer.predict_one(2.4715629565996924)
    0
    >>> scorer.score_one(2.4715629565996924)
    0.99735

    Anomaly is zero due to grace_period
    >>> scorer.predict_one(2.4715629565996924)
    0
    >>> scorer.learn_one(1).gaussian.sigma
    0.5773502691896258
    >>> scorer.predict_one(2.4715629565996924)
    1

    Keeps the sigma due to window_size of 3
    >>> scorer.learn_one(1).gaussian.sigma
    0.5773502691896258
    >>> scorer.process_one(0.5)
    (0, np.float64(2.276441079814074), np.float64(-0.943107746480741))

    Gaussian scorer on time rolling window
    >>> import datetime as dt
    >>> from river.utils import TimeRolling
    >>> scorer = GaussianScorer(
    ...     TimeRolling(Gaussian(),
    ...     period=dt.timedelta(hours=24*7)))
    >>> scorer.process_one(1, t=dt.datetime(2022,2,2))
    (0, np.float64(nan), np.float64(nan))

    Gaussian scorer without window
    >>> scorer = GaussianScorer(Gaussian(), grace_period=2)
    >>> scorer.process_one(1)
    (0, np.float64(nan), np.float64(nan))

    Gaussian scorer with multivariate support. In this case it might be
    practical to specify threshold as lower bound log_threshold for better
    management of low joint likelihood values.
    >>> from river.proba import MultivariateGaussian
    >>> scorer = GaussianScorer(utils.Rolling(MultivariateGaussian(), 2),
    ...     grace_period=1, log_threshold=-8)
    >>> scorer.learn_one({"a": 1, "b": 2}).gaussian.mu
    {'a': 1.0, 'b': 2.0}
    >>> scorer.learn_one({"a": 2, "b": 3}).gaussian.mu
    {'a': 1.5, 'b': 2.5}
    >>> np.log(scorer.score_one({"a": 0, "b": 0}))
    np.float64(-8.49996245328...)
    >>> scorer.predict_one({"a": 0, "b": 0})
    0
    >>> scorer.limit_one()  # doctest: +NORMALIZE_WHITESPACE
    ({'a': np.float64(3.767...), 'b': np.float64(4.767...)},
     {'a': np.float64(-2.160...), 'b': np.float64(-1.160...)})

    Behind the scenes, the threshold is adapted to the dimensionality of the
    input
    >>> np.log(scorer.score_one({"a": -2.161, "b": -1.161}))
    np.float64(-16.000...)
    >>> scorer.predict_one({"a": -2.161, "b": -1.161})
    1
    >>> scorer.predict_one({"a": -2.160, "b": -1.160})
    0

    Known physical bounds complement the learned envelope, so the
    dynamic limits "may be used as an addition to static operating
    limits used by monitoring systems in SCADA" (ESwA 2023). Bounds
    must satisfy low < high
    >>> GaussianScorer(Gaussian(), physical_limits=(2.0, 0.0))
    Traceback (most recent call last):
    ...
    ValueError: physical_limits must satisfy low < high; got (2.0, 0.0)

    Physically impossible observations are flagged even during the
    grace period, and the reported limits are clipped into the bounds
    >>> scorer = GaussianScorer(Rolling(Gaussian(), 3), grace_period=2,
    ...     physical_limits=(0.0, 2.0), protect_anomaly_detector=False)
    >>> scorer.predict_one(5.0)
    1
    >>> scorer = scorer.learn_one(1).learn_one(0)
    >>> scorer.limit_one()
    (np.float64(2.0), np.float64(0.0))

    """

    # The conditional subclass keys the bounds by feature name.
    physical_limits: (
        tuple[float, float] | dict[str, tuple[float, float]] | None
    )

    def __init__(
        self,
        gaussian: Distribution
        | ConditionableDistribution
        | Rolling
        | TimeRolling,
        threshold: float = 0.99735,
        log_threshold: float | None = None,
        grace_period: timedelta | int | None = None,
        t_a: timedelta | int | None = None,
        protect_anomaly_detector: bool = True,
        physical_limits: tuple[float, float] | None = None,
        learn_on_physical_violation: bool = False,
    ) -> None:
        """Initialize GaussianScorer, validating the distribution protocol."""
        if not isinstance(gaussian, (Distribution, ConditionableDistribution)):
            if isinstance(gaussian, (Rolling, TimeRolling)) and isinstance(
                gaussian.obj,
                (Distribution, ConditionableDistribution),
            ):
                pass
            else:
                msg = f"{gaussian} does not satisfy the necessary protocol"
                raise TypeError(
                    msg,
                )
        self.gaussian = gaussian

        if isinstance(gaussian, Rolling):
            self.t_e = gaussian.window_size or 0
        elif isinstance(gaussian, TimeRolling):
            self.t_e = gaussian.period
        else:
            self.t_e = 0
        if grace_period is None:
            self.grace_period = self.t_e
        elif (
            isinstance(self.t_e, int)
            and isinstance(grace_period, int)
            and self.t_e > 0
            and grace_period > self.t_e
        ) or (
            isinstance(self.t_e, timedelta)
            and isinstance(grace_period, timedelta)
            and (grace_period > self.t_e)
        ):
            warnings.warn(
                f"Grace period must be between 1 and "
                f"{self.t_e} minutes or None.",
                stacklevel=2,
            )
            self.grace_period = self.t_e
        elif not isinstance(grace_period, type(self.t_e)):
            msg = (
                "Grace_period must be of the same type as t_e."
                f"Got {type(grace_period)} instead of {type(self.t_e)}."
            )
            raise TypeError(
                msg,
            )
        else:
            self.grace_period = grace_period

        self.threshold = threshold
        self.log_threshold = log_threshold
        if self.log_threshold is not None:
            self.log_threshold_top = np.log1p(-np.exp(self.log_threshold))

        if physical_limits is not None:
            phys_low, phys_high = physical_limits
            if not phys_low < phys_high:
                msg = (
                    "physical_limits must satisfy low < high; "
                    f"got {physical_limits}"
                )
                raise ValueError(msg)
        self.physical_limits = physical_limits
        self.learn_on_physical_violation = learn_on_physical_violation

        self.protect_anomaly_detector = protect_anomaly_detector
        if self.protect_anomaly_detector:
            self.t_a = t_a or self.t_e
            if isinstance(self.t_a, int):
                self.buffer = collections.deque(maxlen=round(self.t_a))
            if isinstance(self.t_a, timedelta):
                self.buffer = TimeRollingBuffer(Store(), period=self.t_a)

    def _get_feature_dim_in(self, x: float | dict[str, float]) -> None:
        if not hasattr(self, "_feature_dim_in"):
            if hasattr(x, "__len__"):
                self._feature_dim_in: int = len(cast("Sized", x))
            else:
                self._feature_dim_in = 1

    def _get_feature_names_in(self, x: float | dict[str, float]) -> None:
        if not hasattr(self, "feature_names_in_") and isinstance(x, dict):
            self.feature_names_in_ = sorted(x.keys())

    def _learn_one(
        self,
        x: float | dict[str, float],
        **kwargs: datetime | float | None,
    ) -> Self:
        if not hasattr(self, "feature_names_in_") and isinstance(x, dict):
            self._get_feature_names_in(x)
        if not hasattr(self, "_feature_dim_in"):
            self._get_feature_dim_in(x)
        cast("Rolling", self.gaussian).update(x, **kwargs)
        return self

    def _drift_detected(self) -> bool:
        len_ = len(self.buffer)
        if len_ > 0:
            return sum(self.buffer) / len_ > self.threshold
        return False

    def _physical_violation(self, x: float | dict[str, float]) -> bool:
        # getattr keeps models recovered from pre-physical-limits
        # pickles working; the isinstance guard also excludes the
        # dict-keyed bounds of the conditional subclass, which
        # overrides this method.
        limits = getattr(self, "physical_limits", None)
        if not isinstance(limits, tuple):
            return False
        phys_low, phys_high = limits
        values = x.values() if isinstance(x, dict) else [x]
        return any(not phys_low <= v <= phys_high for v in values)

    def _rejects_learning(self, x: float | dict[str, float]) -> bool:
        return (
            self._physical_violation(x)
            and not self.learn_on_physical_violation
        )

    @property
    def drift_detected(self) -> bool:
        """Whether the recent anomaly rate indicates a regime change.

        Public wrapper over the internal drift signal — the paper's
        changepoint ("regime change") diagnostic: ``True`` when the
        share of anomalies in the protection buffer exceeds the
        threshold. Always ``False`` when ``protect_anomaly_detector``
        is disabled, since no buffer is maintained then.

        Examples:
        --------
        >>> from river.proba import Gaussian
        >>> from river.utils import Rolling
        >>> scorer = GaussianScorer(Rolling(Gaussian(), 3),
        ...     grace_period=2)
        >>> for x in [1.0, 1.1, 0.9]:
        ...     scorer = scorer.learn_one(x)
        >>> scorer.drift_detected
        False

        A persistent level shift fills the buffer with anomalies and is
        reported as drift
        >>> for x in [10.0, 10.0, 10.0]:
        ...     scorer = scorer.learn_one(x)
        >>> scorer.drift_detected
        True
        """
        if not self.protect_anomaly_detector:
            return False
        return self._drift_detected()

    def n_seen(self) -> timedelta | float:
        """Return the number of observations seen, as count or timedelta."""
        if isinstance(self.grace_period, timedelta) and isinstance(
            self.gaussian,
            TimeRolling,
        ):
            timestamps = self.gaussian._timestamps
            if len(timestamps) == 0:
                n_seen = timedelta(0)
            else:
                n_seen = timestamps[-1] - timestamps[0]
        else:
            n_seen = self.gaussian.n_samples
        return n_seen

    # river's base annotates learn_one -> None; ours returns self.
    # Annotating violates Liskov under ty.
    def learn_one(  # noqa: ANN201
        self,
        x: float | dict[str, float],
        **learn_kwargs: datetime | float | None,
    ):
        """Update distribution, skipping anomalous samples when protected."""
        if self._rejects_learning(x):
            # Physically impossible samples are kept out of both the
            # distribution and the protection buffer: a sensor fault
            # must not drive drift adaptation.
            return self
        if self.protect_anomaly_detector:
            is_anomaly = self.predict_one(x)
            self.buffer.append(is_anomaly)
            is_change = self._drift_detected()
            if not is_anomaly or is_change:
                self._learn_one(x, **learn_kwargs)
        else:
            self._learn_one(x, **learn_kwargs)
        return self

    def score_one(self, x: float | dict[str, float]) -> float:
        """Return the CDF anomaly score for x; 0.5 during grace period.

        Repeated calls are reproducible as long as the distribution is not
        updated in between and, for multivariate distributions, a seed is
        set (scipy evaluates the multivariate normal CDF with randomized
        quasi-Monte-Carlo integration).
        """
        if cast("float", self.n_seen()) >= cast("float", self.grace_period):
            return self.gaussian.cdf(cast("dict[str, float]", x))
        if not hasattr(self, "_feature_dim_in"):
            return 0.5
        return 0.5**self._feature_dim_in

    def predict_one(self, x: float | dict[str, float]) -> int:
        """Return 1 if x is anomalous under the threshold, else 0.

        Observations outside the configured ``physical_limits`` are
        flagged unconditionally, even during the grace period.
        """
        self._get_feature_dim_in(x)
        self._get_feature_names_in(x)

        if self._physical_violation(x):
            return 1

        score = self.score_one(x)
        if (
            cast("float", self.n_seen()) > cast("float", self.grace_period)
            and self._feature_dim_in
        ):
            if self.log_threshold:
                score = -np.inf if score <= 0 else np.log(score)
                if (
                    score < self.log_threshold * self._feature_dim_in
                ) or self.log_threshold_top * self._feature_dim_in < score:
                    return 1
            elif ((1 - self.threshold) ** self._feature_dim_in > score) or (
                score > self.threshold**self._feature_dim_in
            ):
                return 1
        return 0

    def limit_one(
        self,
        *args: float | dict[str, float],
        diagonal_only: bool = True,
    ) -> tuple[
        float | np.ndarray | dict[str, float],
        float | np.ndarray | dict[str, float],
    ]:
        """Return (upper, lower) Gaussian limits derived from the threshold.

        The limits are the normal quantiles of the configured
        (log-)threshold scaled to the input dimensionality, mirroring the
        decision rule of ``predict_one``. By default they are informative
        envelopes around the fitted distribution that may drift as it
        adapts; configuring ``physical_limits`` makes them strict process
        boundaries by clipping the reported thresholds into the known
        bounds, so they "may be used as an addition to static operating
        limits used by monitoring systems in SCADA" (ESwA 2023).
        """
        if len(args) > 0:
            self._get_feature_dim_in(args[0])
            self._get_feature_names_in(args[0])

        kwargs = {
            "loc": [*self.gaussian.mu.values()]
            if isinstance(self.gaussian.mu, dict)
            else self.gaussian.mu,
            "scale": self.gaussian.sigma,
        }
        if diagonal_only and isinstance(kwargs["scale"], pd.DataFrame):
            kwargs["scale"] = [
                kwargs["scale"][i][i] for i in kwargs["scale"].columns
            ]
        if not hasattr(self, "_feature_dim_in"):
            _feature_dim_in = 1
        else:
            _feature_dim_in = self._feature_dim_in
        if self.log_threshold:
            thresh_high = norm.ppf(
                np.exp(self.log_threshold_top * _feature_dim_in),
                **kwargs,
            )
            thresh_low = norm.ppf(
                np.exp(self.log_threshold * _feature_dim_in),
                **kwargs,
            )
        else:
            thresh_high = norm.ppf(self.threshold**_feature_dim_in, **kwargs)
            thresh_low = norm.ppf(
                (1 - self.threshold) ** _feature_dim_in,
                **kwargs,
            )
        if (
            hasattr(self, "feature_names_in_")
            and isinstance(self.gaussian.mu, dict)
            and len(thresh_high) == len(self.feature_names_in_)
        ):
            thresh_high = dict(
                zip(self.gaussian.mu.keys(), thresh_high, strict=False),
            )
            thresh_low = dict(
                zip(self.gaussian.mu.keys(), thresh_low, strict=False),
            )
        elif hasattr(self, "feature_names_in_"):
            thresh_high = dict(
                zip(
                    self.feature_names_in_,
                    [np.nan] * self._feature_dim_in,
                    strict=False,
                ),
            )
            thresh_low = dict(
                zip(
                    self.feature_names_in_,
                    [np.nan] * self._feature_dim_in,
                    strict=False,
                ),
            )
        limits = getattr(self, "physical_limits", None)
        if isinstance(limits, tuple):
            phys_low, phys_high = limits
            if isinstance(thresh_high, dict) and isinstance(
                thresh_low,
                dict,
            ):
                thresh_high = {
                    k: np.clip(v, phys_low, phys_high)
                    for k, v in thresh_high.items()
                }
                thresh_low = {
                    k: np.clip(v, phys_low, phys_high)
                    for k, v in thresh_low.items()
                }
            else:
                thresh_high = np.clip(
                    cast("float | np.ndarray", thresh_high),
                    phys_low,
                    phys_high,
                )
                thresh_low = np.clip(
                    cast("float | np.ndarray", thresh_low),
                    phys_low,
                    phys_high,
                )
        return thresh_high, thresh_low

    def process_one(
        self,
        x: float | dict[str, float],
        t: datetime | None = None,
    ) -> tuple[
        int,
        float | np.ndarray | dict[str, float],
        float | np.ndarray | dict[str, float],
    ]:
        """Predict, compute limits, and learn from x in one step."""
        if self.gaussian.n_samples == 0 and not self._rejects_learning(x):
            if isinstance(self.gaussian, (Rolling, TimeRolling)):
                if hasattr(self.gaussian.obj, "_from_state"):
                    self.gaussian.obj = self.gaussian.obj._from_state(  # type: ignore
                        1,
                        x,
                        0,
                        1,
                    )
            elif hasattr(self.gaussian, "_from_state"):
                self.gaussian = self.gaussian._from_state(  # type: ignore
                    1,
                    x,
                    0,
                    1,
                )

        is_anomaly = self.predict_one(x)

        thresh_high, thresh_low = self.limit_one(x)

        if not is_anomaly:
            if isinstance(self.gaussian, utils.TimeRolling):
                self.learn_one(x, t=t)
            else:
                self.learn_one(x)

        return is_anomaly, thresh_high, thresh_low


class ConditionalGaussianScorer(GaussianScorer):
    """Conditional Gaussian Scorer for anomaly detection.

    Parameters
    ----------
        threshold (float): Anomaly threshold.
        window_size (int or None): Size of the rolling window.
        period (int or None): Time period for time rolling.
        grace_period (int): Grace period before scoring starts.
        physical_limits (dict[str, tuple[float, float]] or None): Known
        static (low, high) bounds keyed by feature name. Reported
        dynamic limits are clipped into them per feature, and a
        violation forces an anomaly with the violated feature as root
        cause, even during the grace period.
        learn_on_physical_violation (bool): Whether physically
        impossible samples still update the distribution. Defaults to
        False, excluding them from learning entirely.

    Examples:
    --------
    Make sure that the passed distribution sattisfies necessary protocol
    >>> bad_scorer = ConditionalGaussianScorer(
    ...     type('Dist', (object,), {})(), grace_period=0, t_a=0
    ...     )
    Traceback (most recent call last):
    ...
    TypeError: ... does not satisfy the necessary protocol

    Gaussian scorer on rolling window
    >>> from river.utils import Rolling
    >>> from functions.proba import MultivariateGaussian
    >>> scorer = ConditionalGaussianScorer(Rolling(MultivariateGaussian(), 3),
    ...     grace_period=1, protect_anomaly_detector=False)

    Initial values
    >>> scorer.gaussian.mu
    {}
    >>> scorer.limit_one({"a": 1, "b": 2})
    ({'a': nan, 'b': nan}, {'a': nan, 'b': nan})

    During grace period, the score is kept 0.5 and prediction is 0
    >>> scorer.learn_one({"a": 1.5, "b": 0.5}).gaussian.mu
    {'a': 1.5, 'b': 0.5}
    >>> scorer.score_one({"a": 1, "b": 2})
    0.5
    >>> scorer.predict_one({"a": 1, "b": 2})
    0
    >>> scorer.limit_one({"a": 1, "b": 2})  # doctest: +NORMALIZE_WHITESPACE
    ({'a': np.float64(1.5), 'b': np.float64(0.5)},
     {'a': np.float64(1.5), 'b': np.float64(0.5)})

    Let's learn some more samples
    >>> scorer.learn_one({"a": 1., "b": 2.}).gaussian.mu
    {'a': 1.25, 'b': 1.25}
    >>> scorer.learn_one({"a": 0.5, "b": 2.}).gaussian.mu
    {'a': 1.0, 'b': 1.5}
    >>> scorer.gaussian.var
           a      b
    a  0.250 -0.375
    b -0.375  0.750
    >>> scorer.score_one({"a": 1., "b": 1.5})
    np.float64(0.5)
    >>> scorer.score_one({"a": 1., "b": 2.})
    np.float64(0.875...)
    >>> scorer.limit_one({"a": 1., "b": 2.})  # doctest: +NORMALIZE_WHITESPACE
    ({'a': np.float64(1.501...), 'b': np.float64(2.801...)},
     {'a': np.float64(-0.001...), 'b': np.float64(0.198...)})
    >>> scorer.limit_one({"b": 2., "a": 1.})  # doctest: +NORMALIZE_WHITESPACE
    ({'a': np.float64(1.501...), 'b': np.float64(2.801...)},
     {'a': np.float64(-0.001...), 'b': np.float64(0.198...)})
    >>> scorer.predict_one({"a": 1.0, "b": 2.802})
    1
    >>> scorer.get_root_cause()
    'b'
    >>> scorer.score_one({"a": 1.0, "b": 2.802})
    np.float64(0.998...)
    >>> scorer.predict_one({"a": 1.0, "b": 2.801})
    0
    >>> scorer.score_one({"a": 1.0, "b": 2.801})
    np.float64(0.99867...)

    Per-feature physical bounds clip the dynamic limits and force
    anomalies, so the limits "may be used as an addition to static
    operating limits used by monitoring systems in SCADA" (ESwA 2023).
    Violations are flagged even during the grace period, with the
    violated feature as root cause
    >>> scorer = ConditionalGaussianScorer(Rolling(MultivariateGaussian(), 3),
    ...     grace_period=1, protect_anomaly_detector=False,
    ...     physical_limits={"b": (0.0, 2.5)})
    >>> scorer.predict_one({"a": 1.0, "b": 9.9})
    1
    >>> scorer.get_root_cause()
    'b'
    >>> for x in [{"a": 1.5, "b": 0.5}, {"a": 1., "b": 2.},
    ...           {"a": 0.5, "b": 2.}]:
    ...     scorer = scorer.learn_one(x)
    >>> scorer.limit_one({"a": 1., "b": 2.})  # doctest: +NORMALIZE_WHITESPACE
    ({'a': np.float64(1.501...), 'b': np.float64(2.5)},
     {'a': np.float64(-0.001...), 'b': np.float64(0.198...)})

    """

    gaussian: ConditionableDistribution | Rolling | TimeRolling

    def __init__(
        self,
        gaussian: ConditionableDistribution | Rolling | TimeRolling,
        threshold: float = 0.99735,
        grace_period: timedelta | int | None = None,
        t_a: timedelta | int | None = None,
        protect_anomaly_detector: bool = True,
        physical_limits: dict[str, tuple[float, float]] | None = None,
        learn_on_physical_violation: bool = False,
    ) -> None:
        """Initialize ConditionalGaussianScorer with a conditionable dist."""
        if not isinstance(gaussian, ConditionableDistribution):
            if isinstance(gaussian, (Rolling, TimeRolling)) and isinstance(
                gaussian.obj,
                ConditionableDistribution,
            ):
                pass
            else:
                msg = f"{gaussian} does not satisfy the necessary protocol"
                raise TypeError(
                    msg,
                )
        super().__init__(
            gaussian=gaussian,
            threshold=threshold,
            grace_period=grace_period,
            t_a=t_a,
            protect_anomaly_detector=protect_anomaly_detector,
            learn_on_physical_violation=learn_on_physical_violation,
        )
        if physical_limits is not None:
            for name, (phys_low, phys_high) in physical_limits.items():
                if not phys_low < phys_high:
                    msg = (
                        "physical_limits must satisfy low < high; got "
                        f"{(phys_low, phys_high)} for feature {name!r}"
                    )
                    raise ValueError(msg)
        self.physical_limits = physical_limits
        self.gaussian = gaussian
        self.root_cause = None
        self.alpha = (1 - threshold) / 2

    def _physical_violations(self, x: dict[str, float]) -> list[str]:
        # Violated features sorted by how far they exceed their bound,
        # so the worst offender leads. getattr keeps models recovered
        # from pre-physical-limits pickles working.
        limits = getattr(self, "physical_limits", None)
        if not isinstance(limits, dict):
            return []
        violated = [
            name
            for name, (phys_low, phys_high) in limits.items()
            if name in x and not phys_low <= x[name] <= phys_high
        ]
        violated.sort(
            key=lambda name: max(
                limits[name][0] - x[name],
                x[name] - limits[name][1],
            ),
            reverse=True,
        )
        return violated

    def _physical_violation(self, x: float | dict[str, float]) -> bool:
        if not isinstance(x, dict):
            return False
        return bool(self._physical_violations(x))

    def _farthest_from_center(
        self,
        input_list: list[float],
    ) -> tuple[float | None, int | None]:
        # Initialize variables to keep track of the farthest element and its
        #  difference
        farthest_element = None
        farthest_index = None
        max_difference = float("-inf")

        for index, value in enumerate(input_list):
            # Calculate the abs difference between the current value and 0.5
            difference = abs(value - 0.5)

            # Check if the current difference is greater than the current
            #  maximum difference
            if difference > max_difference:
                farthest_element = value
                farthest_index = index
                max_difference = difference

        return farthest_element, farthest_index

    def _scores_one(self, x: dict[str, float]) -> list:
        scores = []
        cg = cast("ConditionableDistribution", self.gaussian)
        mean = cg.mu
        covariance = cg.var
        for var_key, var_val in x.items():
            cond_mean, _, cond_std = cg.mv_conditional(
                x,
                var_key,
                mean,
                covariance,
            )
            if cond_std[0] > 0:
                scores.append(
                    norm.cdf(var_val, loc=cond_mean[0], scale=cond_std[0]),
                )
            else:
                scores.append(0.0)
        return scores

    def _score_one(
        self,
        x: float | dict[str, float],
    ) -> tuple[float, int | None]:
        if not self.grace_period or cast("float", self.n_seen()) > cast(
            "float",
            self.grace_period,
        ):
            # Deactivate grace period after first invocation
            self.grace_period = None
            scores = self._scores_one(cast("dict[str, float]", x))
            score, idx = self._farthest_from_center(scores)
            if score is None:
                # No conditional score could be computed (no features in
                # x); report the neutral score instead of flagging an
                # anomaly.
                return 0.5, None
            return score, idx
        return 0.5, None

    def scores_one(self, x: dict[str, float]) -> dict[str, float]:
        """Return per-signal conditional CDF scores keyed by feature name.

        Each signal is scored under its Gaussian distribution
        conditioned on the remaining observed signals — the paper's
        root-cause isolation diagnostic. Returns the neutral score 0.5
        for every feature until the covariance estimate is defined.

        Examples:
        --------
        >>> from river.utils import Rolling
        >>> from functions.proba import MultivariateGaussian
        >>> scorer = ConditionalGaussianScorer(
        ...     Rolling(MultivariateGaussian(), 3),
        ...     grace_period=1, protect_anomaly_detector=False)
        >>> scorer.scores_one({"a": 1., "b": 2.})
        {'a': 0.5, 'b': 0.5}
        >>> for x in [{"a": 1.5, "b": 0.5}, {"a": 1., "b": 2.},
        ...           {"a": 0.5, "b": 2.}]:
        ...     scorer = scorer.learn_one(x)
        >>> scorer.scores_one({"a": 1., "b": 2.})
        {'a': np.float64(0.841...), 'b': np.float64(0.875...)}
        """
        cg = cast("ConditionableDistribution", self.gaussian)
        if cg.var.shape[0] == 0:
            return dict.fromkeys(x, 0.5)
        return dict(zip(x, self._scores_one(x), strict=True))

    def rank_root_causes(
        self,
        x: dict[str, float],
        k: int | None = None,
    ) -> list[str]:
        """Return features ranked as root-cause candidates for ``x``.

        Features are sorted by the deviation of their per-signal
        conditional score from the neutral 0.5 — the same criterion as
        the internal ``_farthest_from_center`` argmax behind
        ``get_root_cause`` — extending the paper's root-cause isolation
        from a single signal to a ranked top-``k``.

        Args:
            x: Observation keyed by feature name.
            k: Number of top candidates to return; ``None`` returns
                all features.

        Returns:
            Feature names sorted by decreasing deviation from center.

        Examples:
        --------
        >>> from river.utils import Rolling
        >>> from functions.proba import MultivariateGaussian
        >>> scorer = ConditionalGaussianScorer(
        ...     Rolling(MultivariateGaussian(), 3),
        ...     grace_period=1, protect_anomaly_detector=False)
        >>> for x in [{"a": 1.5, "b": 0.5}, {"a": 1., "b": 2.},
        ...           {"a": 0.5, "b": 2.}]:
        ...     scorer = scorer.learn_one(x)
        >>> scorer.rank_root_causes({"a": 1., "b": 2.})
        ['b', 'a']
        >>> scorer.rank_root_causes({"a": 1., "b": 2.}, k=1)
        ['b']
        """
        scores = self.scores_one(x)
        ranked = sorted(
            scores,
            key=lambda name: abs(scores[name] - 0.5),
            reverse=True,
        )
        return ranked if k is None else ranked[:k]

    def get_root_cause(self) -> str | int | None:
        """Return feature name identified as root cause of the last anomaly."""
        return self.root_cause

    def score_one(self, x: float | dict[str, float]) -> float:
        """Return the conditional anomaly score farthest from 0.5.

        Returns the neutral score 0.5 while the grace period has not
        elapsed, and also when no conditional score can be computed
        (e.g. ``x`` carries no features).
        """
        score, _ = self._score_one(x)
        return score

    def predict_one(self, x: float | dict[str, float]) -> int:
        """Return 1 and set root cause if x is anomalous, else 0.

        A violation of the configured ``physical_limits`` forces an
        anomaly — even during the grace period — and attributes the
        most violated feature as root cause.
        """
        self._get_feature_dim_in(x)
        self._get_feature_names_in(x)

        if isinstance(x, dict):
            violations = self._physical_violations(x)
            if violations:
                self.root_cause = violations[0]
                return 1

        score, idx = self._score_one(x)
        if (self.alpha > score) or (score > 1 - self.alpha):
            if hasattr(self, "feature_names_in_") and idx is not None:
                self.root_cause = self.feature_names_in_[idx]
            elif idx is not None:
                self.root_cause = idx
            else:
                self.root_cause = None
            return 1
        self.root_cause = None
        return 0

    def _get_limits(
        self,
        c_mean: np.ndarray,
        c_std: np.ndarray,
    ) -> tuple[float, float]:
        z_critical = norm.ppf(1 - self.alpha)

        lower_bound = c_mean - z_critical * c_std
        upper_bound = c_mean + z_critical * c_std

        return lower_bound[0], upper_bound[0]

    def limit_one(  # type: ignore[override]
        self,
        x: dict[str, float] | None = None,
        # gradual varargs required for ty override-compat with parent
        *_args,  # noqa: ANN002
        **_kwargs,  # noqa: ANN003
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return per-feature (upper, lower) conditional limits.

        Safe to call before any ``learn_one``/``predict_one``: feature
        names are inferred from ``x`` and NaN limits are returned until
        the covariance estimate is defined. Features with configured
        ``physical_limits`` have their reported limits clipped into the
        known static bounds.
        """
        if x is None:
            x = {}
        self._get_feature_dim_in(x)
        self._get_feature_names_in(x)

        ths = dict.fromkeys(self.feature_names_in_, np.nan)
        tls = ths.copy()
        cg = cast("ConditionableDistribution", self.gaussian)
        if cg.var.shape[0] != 0:
            for var_key in self.feature_names_in_:
                cond_mean, _, cond_std = cg.mv_conditional(
                    x,
                    var_key,
                    cg.mu,
                    cg.var,
                )
                tls[var_key], ths[var_key] = self._get_limits(
                    cond_mean,
                    cond_std,
                )

        limits = getattr(self, "physical_limits", None)
        if isinstance(limits, dict):
            for name, (phys_low, phys_high) in limits.items():
                if name in ths:
                    ths[name] = np.clip(ths[name], phys_low, phys_high)
                    tls[name] = np.clip(tls[name], phys_low, phys_high)

        return ths, tls

    def get_limits(
        self,
        x: dict[str, float] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Return per-signal (lower, upper) limits keyed by feature name.

        Public view of the dynamic operating limits computed from the
        conditional moments — the paper's dynamic signal limits
        diagnostic. The values agree with ``limit_one``, which returns
        the same limits grouped as (upper, lower) dicts instead, and
        are clipped into any configured per-feature ``physical_limits``.

        Examples:
        --------
        >>> from river.utils import Rolling
        >>> from functions.proba import MultivariateGaussian
        >>> scorer = ConditionalGaussianScorer(
        ...     Rolling(MultivariateGaussian(), 3),
        ...     grace_period=1, protect_anomaly_detector=False)
        >>> for x in [{"a": 1.5, "b": 0.5}, {"a": 1., "b": 2.},
        ...           {"a": 0.5, "b": 2.}]:
        ...     scorer = scorer.learn_one(x)
        >>> scorer.get_limits({"a": 1., "b": 2.})
        {'a': (np.float64(-0.001...), np.float64(1.501...)),
         'b': (np.float64(0.198...), np.float64(2.801...))}
        """
        ths, tls = self.limit_one(x)
        return {key: (tls[key], ths[key]) for key in ths}
