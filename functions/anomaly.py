import collections
import warnings
from datetime import timedelta
from typing import Protocol, Tuple, Union, runtime_checkable

import numpy as np
import pandas as pd
from river import anomaly, utils
from river.utils import Rolling, TimeRolling
from scipy.stats import norm


@runtime_checkable
class Distribution(Protocol):  # pragma: no cover
    mu: Union[float, dict[str, float]]
    sigma: Union[float, pd.DataFrame]
    n_samples: Union[float, int]

    def update(self, *args, **kwargs):
        ...

    def cdf(self, *args, **kwargs) -> float:
        ...


@runtime_checkable
class ConditionableDistribution(
    Distribution, Protocol
):  # pragma: no cover  # noqa: E501
    mu: dict[str, float]
    sigma: pd.DataFrame
    var: pd.DataFrame
    n_samples: Union[float, int]

    def update(self, *args, **kwargs):
        ...

    def cdf(self, *args, **kwargs) -> float:
        ...

    def mv_conditional(
        self, *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


class Store:
    """
    A custom store that implements basic list-like functionality.

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

    def __init__(self):
        self.x: list = []

    def __iter__(self):
        yield from self.x

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)

    def append(self, *args, **kwargs):
        self.x.append(args[0])

    def update(self, *args, **kwargs):
        self.x.append(args[0])

    def revert(self, *args, **kwargs):
        self.x.pop(0)


# TODO: Find a better way to expose __len__ of parent class
TimeRolling.__len__ = lambda self: len(self.x)  # type: ignore


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

    Examples
    --------
    Make sure that the passed distribution sattisfies necessary protocol
    >>> bad_scorer = GaussianScorer(
    ...     type('Dist', (object,), {})(), grace_period=0
    ...     )
    Traceback (most recent call last):
    ...
    ValueError:  does not satisfy the necessary protocol

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
    (nan, nan)
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
    (0, 2.276441079814074, -0.943107746480741)

    Gaussian scorer on time rolling window
    >>> import datetime as dt
    >>> from river.utils import TimeRolling
    >>> scorer = GaussianScorer(
    ...     TimeRolling(Gaussian(),
    ...     period=dt.timedelta(hours=24*7)))
    >>> scorer.process_one(1, t=dt.datetime(2022,2,2))
    (0, nan, nan)

    Gaussian scorer without window
    >>> scorer = GaussianScorer(Gaussian(), grace_period=2)
    >>> scorer.process_one(1)
    (0, nan, nan)

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
    -8.4999624532873
    >>> scorer.predict_one({"a": 0, "b": 0})
    0
    >>> scorer.limit_one()
    ({'a': 3.767..., 'b': 4.767...}, {'a': -2.160..., 'b': -1.160...})

    Behind the scenes, the threshold is adapted to the dimensionality of the
    input
    >>> np.log(scorer.score_one({"a": -2.161, "b": -1.161}))
    -16.000...
    >>> scorer.predict_one({"a": -2.161, "b": -1.161})
    1
    >>> scorer.predict_one({"a": -2.160, "b": -1.160})
    0
    """  # noqa: E501

    def __init__(
        self,
        gaussian: Union[
            Distribution, ConditionableDistribution, Rolling, TimeRolling
        ],
        threshold: float = 0.99735,
        log_threshold: Union[float, None] = None,
        grace_period: Union[timedelta, int, None] = None,
        t_a: Union[timedelta, int, None] = None,
        protect_anomaly_detector: bool = True,
    ):
        if not isinstance(gaussian, (Distribution, ConditionableDistribution)):
            if isinstance(gaussian, (Rolling, TimeRolling)) and isinstance(
                gaussian.obj, (Distribution, ConditionableDistribution)
            ):
                pass
            else:
                raise ValueError(
                    f"{gaussian} does not satisfy the necessary protocol"
                )
        self.gaussian = gaussian

        if isinstance(gaussian, Rolling):
            self.t_e = gaussian.window_size if gaussian.window_size else 0
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
                f"{self.t_e} minutes or None."
            )
            self.grace_period = self.t_e
        elif not isinstance(grace_period, type(self.t_e)):
            raise TypeError(
                "Grace_period must be of the same type as t_e."
                f"Got {type(grace_period)} instead of {type(self.t_e)}."
            )
        else:
            self.grace_period = grace_period

        self.threshold = threshold
        self.log_threshold = log_threshold
        if self.log_threshold is not None:
            self.log_threshold_top = np.log1p(-np.exp(self.log_threshold))

        self.protect_anomaly_detector = protect_anomaly_detector
        if self.protect_anomaly_detector:
            self.t_a = t_a if t_a else self.t_e
            if isinstance(self.t_a, int):
                self.buffer = collections.deque(maxlen=round(self.t_a))
            if isinstance(self.t_a, timedelta):
                self.buffer = TimeRolling(Store(), period=self.t_a)

    def _get_feature_dim_in(self, x):
        if not hasattr(self, "_feature_dim_in"):
            if hasattr(x, "__len__"):
                self._feature_dim_in: int = len(x)
            else:
                self._feature_dim_in = 1

    def _get_feature_names_in(self, x: dict[str, float]):
        if not hasattr(self, "feature_names_in_") and isinstance(x, dict):
            self.feature_names_in_ = sorted(list(x.keys()))

    def _learn_one(self, x, **kwargs):
        if not hasattr(self, "feature_names_in_") and isinstance(x, dict):
            self._get_feature_names_in(x)
        if not hasattr(self, "_feature_dim_in"):
            self._get_feature_dim_in(x)
        self.gaussian.update(x, **kwargs)
        return self

    def _drift_detected(self) -> bool:
        len_ = len(self.buffer)  # type: ignore
        if len_ > 0:
            # return sum(self.buffer) / len_ > 1 - self.alpha
            return sum(self.buffer) / len_ > (self.threshold)  # type: ignore
        else:
            return False

    def n_seen(self):
        if isinstance(self.grace_period, timedelta) and isinstance(
            self.gaussian, TimeRolling
        ):
            # TODO: remove this statement after river ~= 0.20.0 is buildable
            if hasattr(self.gaussian, "_timestamps"):
                timestamps = self.gaussian._timestamps
            else:
                timestamps = [event[0] for event in self.gaussian._events]
            if len(timestamps) == 0:
                n_seen = timedelta(0)
            else:
                n_seen = timestamps[-1] - timestamps[0]
        else:
            n_seen = self.gaussian.n_samples
        return n_seen

    def learn_one(self, x, **learn_kwargs):
        if self.protect_anomaly_detector:
            is_anomaly = self.predict_one(x)
            self.buffer.append(is_anomaly)
            is_change = self._drift_detected()
            if not is_anomaly or is_change:
                self._learn_one(x, **learn_kwargs)
        else:
            self._learn_one(x, **learn_kwargs)
        return self

    def score_one(self, x) -> float:
        # TODO: find out why return different results on each invocation
        if self.n_seen() >= self.grace_period:  # type: ignore
            return self.gaussian.cdf(x)
        if not hasattr(self, "_feature_dim_in"):
            return 0.5
        else:
            return 0.5**self._feature_dim_in

    def predict_one(self, x) -> int:
        self._get_feature_dim_in(x)
        self._get_feature_names_in(x)

        score = self.score_one(x)
        if (
            self.n_seen() > self.grace_period  # type: ignore
            and self._feature_dim_in
        ):
            if self.log_threshold:
                score = -np.inf if score <= 0 else np.log(score)
                if (
                    score < self.log_threshold * self._feature_dim_in
                ) or self.log_threshold_top * self._feature_dim_in < score:
                    return 1
            else:
                if ((1 - self.threshold) ** self._feature_dim_in > score) or (
                    score > self.threshold**self._feature_dim_in
                ):
                    return 1
        return 0

    def limit_one(self, *args, diagonal_only=True):
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
        # TODO: consider strict process boundaries
        # real_thresh = norm.ppf((self.sigma/2 + 0.5), **kwargs)
        # TODO: following code changes the limits given by former
        if not hasattr(self, "_feature_dim_in"):
            _feature_dim_in = 1
        else:
            _feature_dim_in = self._feature_dim_in
        if self.log_threshold:
            thresh_high = norm.ppf(
                np.exp(self.log_threshold_top * _feature_dim_in), **kwargs
            )
            thresh_low = norm.ppf(
                np.exp(self.log_threshold * _feature_dim_in), **kwargs
            )
        else:
            thresh_high = norm.ppf(self.threshold**_feature_dim_in, **kwargs)
            thresh_low = norm.ppf(
                (1 - self.threshold) ** _feature_dim_in, **kwargs
            )
        if (
            hasattr(self, "feature_names_in_")
            and isinstance(self.gaussian.mu, dict)
            and len(thresh_high) == len(self.feature_names_in_)
        ):
            thresh_high = dict(zip(self.gaussian.mu.keys(), thresh_high))
            thresh_low = dict(zip(self.gaussian.mu.keys(), thresh_low))
        elif hasattr(self, "feature_names_in_"):
            thresh_high = dict(
                zip(self.feature_names_in_, [np.nan] * self._feature_dim_in)
            )
            thresh_low = dict(
                zip(self.feature_names_in_, [np.nan] * self._feature_dim_in)
            )
        return thresh_high, thresh_low

    def process_one(self, x, t=None):
        if self.gaussian.n_samples == 0:
            if isinstance(self.gaussian, (Rolling, TimeRolling)):
                if hasattr(self.gaussian.obj, "_from_state"):
                    self.gaussian.obj = (  # type: ignore
                        self.gaussian.obj._from_state(  # type: ignore
                            1, x, 0, 1
                        )
                    )
            else:
                if hasattr(self.gaussian, "_from_state"):
                    self.gaussian = self.gaussian._from_state(  # type: ignore
                        1, x, 0, 1
                    )

        is_anomaly = self.predict_one(x)

        thresh_high, thresh_low = self.limit_one(x)

        if not is_anomaly:
            if isinstance(self.gaussian, utils.TimeRolling):
                self = self.learn_one(x, **{"t": t})
            else:
                self = self.learn_one(x)

        return is_anomaly, thresh_high, thresh_low


class ConditionalGaussianScorer(GaussianScorer):
    """Conditional Gaussian Scorer for anomaly detection.

    Parameters
    ----------
        threshold (float): Anomaly threshold.
        window_size (int or None): Size of the rolling window.
        period (int or None): Time period for time rolling.
        grace_period (int): Grace period before scoring starts.

    Examples
    --------
    Make sure that the passed distribution sattisfies necessary protocol
    >>> bad_scorer = ConditionalGaussianScorer(
    ...     type('Dist', (object,), {})(), grace_period=0, t_a=0
    ...     )
    Traceback (most recent call last):
    ...
    ValueError:  does not satisfy the necessary protocol

    Gaussian scorer on rolling window
    >>> from river.utils import Rolling
    >>> from proba import MultivariateGaussian
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
    >>> scorer.limit_one({"a": 1, "b": 2})
    ({'a': 1.5, 'b': 0.5}, {'a': 1.5, 'b': 0.5})

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
    0.5
    >>> scorer.score_one({"a": 1., "b": 2.})
    0.875...
    >>> scorer.limit_one({"a": 1., "b": 2.})
    ({'a': 1.501..., 'b': 2.801...}, {'a': -0.001..., 'b': 0.198...})
    >>> scorer.limit_one({"b": 2., "a": 1.})
    ({'a': 1.501..., 'b': 2.801...}, {'a': -0.001..., 'b': 0.198...})
    >>> scorer.predict_one({"a": 1.0, "b": 2.802})
    1
    >>> scorer.get_root_cause()
    'b'
    >>> scorer.score_one({"a": 1.0, "b": 2.802})
    0.998...
    >>> scorer.predict_one({"a": 1.0, "b": 2.801})
    0
    >>> scorer.score_one({"a": 1.0, "b": 2.801})
    0.99867
    """  # noqa: E501

    def __init__(
        self,
        gaussian: Union[ConditionableDistribution, Rolling, TimeRolling],
        threshold: float = 0.99735,
        grace_period: Union[timedelta, int, None] = None,
        t_a: Union[timedelta, int, None] = None,
        protect_anomaly_detector: bool = True,
    ):
        if not isinstance(gaussian, ConditionableDistribution):
            if isinstance(gaussian, (Rolling, TimeRolling)) and isinstance(
                gaussian.obj, ConditionableDistribution
            ):
                pass
            else:
                raise ValueError(
                    f"{gaussian} does not satisfy the necessary protocol"
                )
        super().__init__(
            gaussian=gaussian,
            threshold=threshold,
            grace_period=grace_period,
            t_a=t_a,
            protect_anomaly_detector=protect_anomaly_detector,
        )
        self.gaussian = gaussian
        self.root_cause = None
        self.alpha = (1 - threshold) / 2

    def _farthest_from_center(self, input_list):
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

    def _scores_one(self, x) -> list:
        scores = []
        mean = self.gaussian.mu
        covariance = self.gaussian.var
        for var_key in x:
            cond_mean, _, cond_std = self.gaussian.mv_conditional(
                x, var_key, mean, covariance
            )
            if cond_std[0] > 0:
                scores.append(
                    norm.cdf(x[var_key], loc=cond_mean[0], scale=cond_std[0])
                )
            else:
                scores.append(0.0)
        return scores

    def _score_one(self, x):
        # TODO: find out why return different results on each invocation
        #   Due to scipy's cdf function
        if not self.grace_period or self.n_seen() > self.grace_period:
            # Deactivate grace period after first invocation
            self.grace_period = None
            # TODO: generally score is None when the
            #  conditional covariance is maldefined. This
            #  case should be handled differently.
            scores = self._scores_one(x)
            score, idx = self._farthest_from_center(scores)
            return score if score else 1, idx
        return 0.5, None

    def get_root_cause(self):
        return self.root_cause

    def score_one(self, x) -> float:
        score, _ = self._score_one(x)
        return score

    def predict_one(self, x) -> int:
        self._get_feature_dim_in(x)
        self._get_feature_names_in(x)

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

    def _get_limits(self, c_mean: np.ndarray, c_std: np.ndarray):
        z_critical = norm.ppf(1 - self.alpha)

        lower_bound = c_mean - z_critical * c_std
        upper_bound = c_mean + z_critical * c_std

        return lower_bound[0], upper_bound[0]

    def limit_one(self, x: dict[str, float]):
        # TODO: might break the things up in Pipeline if called before
        #  predict_one or learn_one
        self._get_feature_dim_in(x)
        self._get_feature_names_in(x)

        ths = {key: np.nan for key in self.feature_names_in_}
        tls = ths.copy()
        if self.gaussian.var.shape[0] != 0:
            for var_key in self.feature_names_in_:
                cond_mean, _, cond_std = self.gaussian.mv_conditional(
                    x, var_key, self.gaussian.mu, self.gaussian.var
                )
                tls[var_key], ths[var_key] = self._get_limits(
                    cond_mean, cond_std
                )

        return ths, tls
