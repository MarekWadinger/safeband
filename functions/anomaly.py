import typing

import numpy as np
from river import anomaly
from scipy.stats import norm


# CONSTANTS
LOG_THRESHOLD = -25
THRESHOLD = 0.99735
VAR_SMOOTHING = 1e-9


@typing.runtime_checkable
class Distribution(typing.Protocol):
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
                  "scale": self.gaussian.sigma}
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
            self = self.learn_one(x, **{"t": t})

        return is_anomaly, thresh_high, thresh_low
