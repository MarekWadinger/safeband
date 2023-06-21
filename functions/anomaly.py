import typing

import numpy as np
from river import anomaly, utils
from scipy.stats import norm


# CONSTANTS
LOG_THRESHOLD = -25
THRESHOLD = 0.99735
VAR_SMOOTHING = 1e-9


@typing.runtime_checkable
class Distribution(typing.Protocol):  # pragma: no cover
    mu: typing.Optional[float | typing.Sequence[float]]
    sigma: typing.Optional[float | typing.Sequence[float]]
    n_samples: typing.Optional[float]

    def _from_state(self, *args, **kwargs):
        ...

    def update(self, *args, **kwargs):
        ...

    def cdf(self, *args, **kwargs):
        ...


class GaussianScorer(anomaly.base.SupervisedAnomalyDetector):
    """
    Gaussian Scorer for anomaly detection.

    Args:
        threshold (float): Anomaly threshold.
        window_size (int or None): Size of the rolling window.
        period (int or None): Time period for time rolling.
        grace_period (int): Grace period before scoring starts.

    Examples:
    >>> bad_scorer = GaussianScorer(
    ...     type('Dist', (object,), {})(), grace_period=0
    ...     )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ValueError:  does not satisfy the necessary protocol

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
    >>> scorer.score_log_one(2.4715629565996924)
    0.5
    >>> scorer.learn_one(1).gaussian.mu
    1.0
    >>> scorer.gaussian.sigma
    0.0
    >>> scorer.learn_one(0).gaussian.sigma
    0.7071067811865476
    >>> scorer.limit_one()
    (2.4715629565996924, -1.4715629565996926)
    >>> scorer.predict_one(2.4715629565996924)
    0
    >>> scorer.predict_log_one(2.4715629565996924)
    0
    >>> scorer.score_one(2.4715629565996924)
    0.99735
    >>> scorer.score_log_one(2.4715629565996924)
    -0.002653517465563446

    Anomaly is zero due to grace_period
    >>> scorer.predict_one(2.4715629565996924)
    0
    >>> scorer.learn_one(1).gaussian.sigma
    0.5773502691896258
    >>> scorer.predict_one(2.4715629565996924)
    1
    >>> scorer.predict_log_one(2.4715629565996924)
    0

    Keeps the sigma due to window_size of 3
    >>> scorer.learn_one(1).gaussian.sigma
    0.5773502691896258
    >>> scorer.process_one(0.5)
    (0, 2.276441079814074, -0.943107746480741)

    Gaussian scorer on time rolling window
    >>> import datetime as dt
    >>> from river.utils import TimeRolling
    >>> scorer = GaussianScorer(TimeRolling(Gaussian(), period=dt.timedelta(hours=24*7)),
    ...     grace_period=2)
    >>> scorer.process_one(1, t=dt.datetime(2022,2,2))
    (0, nan, nan)

    Gaussian scorer without window
    >>> scorer = GaussianScorer(Gaussian(), grace_period=2)
    >>> scorer.process_one(1)
    (0, nan, nan)
    """
    def __init__(self,
                 obj: Distribution,
                 grace_period: int,
                 threshold: float = THRESHOLD,
                 log_threshold: float = LOG_THRESHOLD
                 ):
        if not isinstance(obj, Distribution):
            raise ValueError(f"{obj} does not satisfy the necessary protocol")
        self.gaussian = obj
        self.grace_period = grace_period
        self.threshold = threshold
        self.log_threshold = log_threshold

    def learn_one(self, x, **kwargs):
        self.gaussian.update(x, **kwargs)
        return self

    def score_one(self, x, t=None):
        if self.gaussian.n_samples < self.grace_period:
            return 0.5
        # return 2 * abs(self.gaussian.cdf(x) - 0.5)
        return self.gaussian.cdf(x)

    def score_log_one(self, x, t=None):
        if self.gaussian.n_samples < self.grace_period:
            return 0.5
        # return 2 * abs(self.gaussian.cdf(x) - 0.5)
        cdf_ = self.gaussian.cdf(x)
        return -np.inf if cdf_ <= 0 else np.log(cdf_)

    def predict_one(self, x, t=None):
        score = self.score_one(x)
        if self.gaussian.n_samples > self.grace_period:
            return 1 if ((1-self.threshold > score) or
                         (score > self.threshold)) else 0
        else:
            return 0

    def predict_log_one(self, x, t=None):
        score = self.score_log_one(x)
        if self.gaussian.n_samples > self.grace_period:
            return 1 if score < self.log_threshold else 0
        else:
            return 0

    def limit_one(self):
        kwargs = {"loc": self.gaussian.mu,
                  "scale": self.gaussian.sigma
                  if not isinstance(self.gaussian.sigma, complex)
                  else 0}
        # TODO: consider strict process boundaries
        # real_thresh = norm.ppf((self.sigma/2 + 0.5), **kwargs)
        # TODO: following code changes the limits given by former
        thresh_high = norm.ppf(self.threshold, **kwargs)
        thresh_low = norm.ppf(1-self.threshold, **kwargs)
        return thresh_high, thresh_low

    def process_one(self, x, t=None):
        if self.gaussian.n_samples == 0:
            self.gaussian.obj = self.gaussian._from_state(0, x,
                                                          VAR_SMOOTHING, 1)

        is_anomaly = self.predict_one(x)

        thresh_high, thresh_low = self.limit_one()

        if not is_anomaly:
            if isinstance(self.gaussian, utils.TimeRolling):
                self = self.learn_one(x, **{"t": t})
            else:
                self = self.learn_one(x)

        return is_anomaly, thresh_high, thresh_low
