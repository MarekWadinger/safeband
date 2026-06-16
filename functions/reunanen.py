"""Streaming autoencoder outlier detector from Reunanen et al. (2020).

Implements the *detection* component (Algorithms 1 and 2) of
Reunanen, N., Räty, T., Jokinen, J. J., Hoyt, T., Culler, D.:
"Unsupervised online detection and prediction of outliers in streams
of sensor data", International Journal of Data Science and Analytics 9,
285-314 (2020), doi:10.1007/s41060-019-00191-3.

The method processes one observation at a time and keeps no window of
past data: a single-hidden-layer sigmoid autoencoder with tied weights
is trained per sample by SGD on the L1 reconstruction cost, and a point
is flagged as an outlier when its reconstruction cost exceeds an
exponentially weighted moving estimate ``mu + k * sigma`` of the cost
distribution.

The prediction component of the paper (Algorithms 3 and 4, a logistic
regression forecasting outliers ``t`` steps ahead) is deliberately not
implemented: it solves a forecasting task that is out of scope for the
point-wise detection benchmarks in this repository, and the paper
treats detection as an independent process.
"""

from typing import Literal

import numpy as np
from river import anomaly

__all__ = ["ReunanenScorer"]


def _sigmoid(a: np.ndarray) -> np.ndarray:
    # Numerically stable logistic function; scaled outliers may produce
    # large pre-activations because scaling limits are frozen (Eq. 15).
    out = np.empty_like(a, dtype=float)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    exp_a = np.exp(a[~pos])
    out[~pos] = exp_a / (1.0 + exp_a)
    return out


class ReunanenScorer(anomaly.base.AnomalyDetector):
    """Online autoencoder outlier detector of Reunanen et al. (2020).

    A tied-weight autoencoder ``z = s(W^T s(W x + b) + b_z)`` with
    sigmoid activations is updated on every observation by one SGD step
    on the L1 reconstruction cost ``r = sum_j |x_j - z_j|`` (Eqs. 9-14
    of the paper). Inputs are min-max scaled with running per-feature
    limits that are *not* updated on detected outliers (Eq. 15). The
    cost distribution is tracked by exponentially weighted moving
    estimates of its mean and variance (Eqs. 17-19), and a point is an
    outlier when ``r > mu + k * sigma`` (Chebyshev bound, Eq. 16).

    Before detection starts, the scorer self-calibrates (Algorithm 1):
    phase 1 consumes points until every feature has a non-degenerate
    scaling range, phase 2 trains the autoencoder until the cost stops
    improving by more than ``decmin`` for ``M * decmin / d`` consecutive
    points (Eq. 20) or ``M`` total calibration points are consumed.
    While calibrating, ``predict_one`` returns ``False``.

    Faithfulness notes (see the paper, Section 3.4):

    * identical consecutive points are skipped entirely (no SGD, no
      EWMA, no limit update) to prevent memorisation;
    * the autoencoder and the EWMA statistics learn on *every*
      non-identical point, including detected outliers - only the
      scaling limits are protected;
    * the learning rate is constant and weights ``W`` are initialised
      uniformly in ``[0, 1)``; biases start at zero.

    Args:
        n_hidden: Number of hidden neurons ``h`` (the paper uses
            ``h < d``; AUROC was insensitive to ``h`` in [2, 10]).
        lr: Constant SGD learning rate ``alpha`` (Eq. 8).
        gamma: EWMA smoothing factor for the cost statistics (Eq. 17).
        k: Multiplier of the EWMA standard deviation in the outlier
            threshold ``mu + k * sigma`` (Eq. 16).
        M: Maximum number of calibration points (Algorithm 1).
        decmin: Minimum meaningful decrease of the reconstruction cost
            during calibration (Algorithm 1).
        seed: Seed for the uniform weight initialisation.

    Examples:
        >>> import math
        >>> scorer = ReunanenScorer(n_hidden=2, M=200, seed=42)
        >>> scorer.predict_one({"a": 0.0, "b": 1.0})
        False
        >>> for i in range(100):
        ...     scorer = scorer.learn_one(
        ...         {"a": math.sin(i / 5), "b": math.cos(i / 5)},
        ...     )
        >>> scorer.calibrated
        True
        >>> scorer.predict_one({"a": 0.0, "b": 1.0})
        False
        >>> scorer.predict_one({"a": 100.0, "b": -100.0})
        True
        >>> scorer.score_one({"a": 0.0, "b": 1.0}) < 1.0
        True

    """

    def __init__(
        self,
        n_hidden: int = 2,
        lr: float = 0.1,
        gamma: float = 0.1,
        k: float = 3.0,
        M: int = 10_000,
        decmin: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Initialise hyperparameters; weights are built on first input."""
        self.n_hidden = n_hidden
        self.lr = lr
        self.gamma = gamma
        self.k = k
        self.M = M
        self.decmin = decmin
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Model parameters, allocated lazily once d is known.
        self.feature_names_in_: list[str] | None = None
        self._w: np.ndarray = np.empty((0, 0))
        self._b: np.ndarray = np.empty(0)
        self._b_z: np.ndarray = np.empty(0)
        self._x_min: np.ndarray = np.empty(0)
        self._x_max: np.ndarray = np.empty(0)

        # EWMA statistics of the reconstruction cost (mu, S; Eqs. 17-18),
        # initialised to zero at calibration start.
        self._mu = 0.0
        self._s = 0.0

        # Calibration state machine (Algorithm 1).
        self._phase: Literal["limits", "train", "detect"] = "limits"
        self._m_left = M
        self._patience_reset = 0.0
        self._patience = 0.0
        self._r_min = float("inf")

        # Last seen raw input; Algorithm 2 initialises it to zero after
        # calibration so that identical consecutive points are skipped.
        self._x_old: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        """Return whether self-calibration (Algorithm 1) has finished."""
        return self._phase == "detect"

    def _initialize(self, x: dict) -> None:
        if self.feature_names_in_ is not None:
            return
        self.feature_names_in_ = sorted(x)
        d = len(self.feature_names_in_)
        self._w = self._rng.uniform(0.0, 1.0, size=(self.n_hidden, d))
        self._b = np.zeros(self.n_hidden)
        self._b_z = np.zeros(d)
        self._x_min = np.full(d, np.inf)
        self._x_max = np.full(d, -np.inf)
        # Patience reset value P_r = M * decmin / d (Eq. 20). Floor at 1
        # so wide inputs (large d) cannot make patience < 1 and end
        # calibration after a single point.
        self._patience_reset = max(1, round(self.M * self.decmin / d))
        self._patience = self._patience_reset

    def _vector(self, x: dict) -> np.ndarray:
        names = self.feature_names_in_ or sorted(x)
        return np.array([float(x[name]) for name in names])

    def _scale(self, v: np.ndarray) -> np.ndarray:
        # Online min-max scaling (Eq. 15). Features whose limits are not
        # yet spread (calibration phase 1) scale to zero; values outside
        # the frozen limits intentionally leave [0, 1].
        span = self._x_max - self._x_min
        valid = span > 0
        safe_span = np.where(valid, span, 1.0)
        scaled = (v - np.where(valid, self._x_min, v)) / safe_span
        return np.where(valid, scaled, 0.0)

    def _forward(self, v_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Encoder y = s(W x + b) (Eq. 9); decoder z = s(W^T y + b_z)
        # (Eq. 10) with tied weights.
        y = _sigmoid(self._w @ v_scaled + self._b)
        z = _sigmoid(self._w.T @ y + self._b_z)
        return y, z

    @staticmethod
    def _cost(v_scaled: np.ndarray, z: np.ndarray) -> float:
        # L1 reconstruction cost r = sum_j |x_j - z_j| (Eq. 11).
        return float(np.sum(np.abs(v_scaled - z)))

    def _gradients(
        self,
        v_scaled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Backpropagation of the L1 loss through the tied-weight sigmoid
        # autoencoder; with tied weights the gradient w.r.t. W is the sum
        # of the encoder and decoder contributions.
        y, z = self._forward(v_scaled)
        delta_z = np.sign(z - v_scaled) * z * (1.0 - z)
        delta_y = (self._w @ delta_z) * y * (1.0 - y)
        grad_w = np.outer(delta_y, v_scaled) + np.outer(y, delta_z)
        return grad_w, delta_y, delta_z

    def _sgd_step(self, v_scaled: np.ndarray) -> None:
        # One constant-learning-rate SGD step (Eqs. 12-14).
        grad_w, grad_b, grad_b_z = self._gradients(v_scaled)
        self._w -= self.lr * grad_w
        self._b -= self.lr * grad_b
        self._b_z -= self.lr * grad_b_z

    def _update_ewma(self, r: float) -> None:
        # EWMA estimates of the cost mean and variance (Eqs. 17-19);
        # updated on every non-identical point, including outliers.
        mu_old = self._mu
        self._mu = (1.0 - self.gamma) * mu_old + self.gamma * r
        self._s = (1.0 - self.gamma) * (
            self._s + self.gamma * (r - mu_old) ** 2
        )

    def _is_outlier(self, r: float) -> bool:
        # Outlier iff r > mu + k * sigma (Eq. 16, Algorithm 2 line 19).
        return r > self._mu + self.k * float(np.sqrt(self._s))

    def _update_limits(self, v: np.ndarray) -> None:
        self._x_min = np.minimum(self._x_min, v)
        self._x_max = np.maximum(self._x_max, v)

    def _calibrate_limits(self, v: np.ndarray) -> None:
        # Algorithm 1, phase 1: spread the scaling limits until every
        # feature has variance.
        self._update_limits(v)
        self._m_left -= 1
        if bool(np.all(self._x_min < self._x_max)):
            self._phase = "train"
        elif self._m_left <= 0:
            # Degenerate stream (a feature never varied) exhausted the
            # calibration budget; start detecting with what we have.
            self._finish_calibration()

    def _calibrate_train(self, v: np.ndarray) -> None:
        # Algorithm 1, phase 2: initial training with a patience
        # heuristic (early stopping).
        self._update_limits(v)
        v_scaled = self._scale(v)
        _, z = self._forward(v_scaled)
        r = self._cost(v_scaled, z)
        self._sgd_step(v_scaled)
        self._update_ewma(r)
        if self._r_min - r > self.decmin:
            self._r_min = r
            self._patience = self._patience_reset
        else:
            self._patience -= 1
        self._m_left -= 1
        if self._patience <= 0 or self._m_left <= 0:
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        self._phase = "detect"
        # Algorithm 2 line 2: x_old <- 0.
        self._x_old = np.zeros_like(self._x_min)

    def _detect_step(self, v: np.ndarray) -> None:
        # Algorithm 2 main loop (lines 3-12).
        if self._x_old is not None and np.array_equal(v, self._x_old):
            # Skip identical consecutive points entirely.
            return
        self._x_old = v
        v_scaled = self._scale(v)
        _, z = self._forward(v_scaled)
        r = self._cost(v_scaled, z)
        if not self._is_outlier(r):
            # Outliers must not contaminate the scaling limits.
            self._update_limits(v)
        self._sgd_step(v_scaled)
        self._update_ewma(r)

    # river's base annotates learn_one -> None; ours returns self.
    # Annotating violates Liskov under ty.
    def learn_one(self, x: dict):  # noqa: ANN201
        """Consume one observation, calibrating or detecting as needed.

        Routes the point through the Algorithm 1 calibration state
        machine (scaling limits, then patience-based initial training)
        and, once calibrated, through the Algorithm 2 online update:
        skip identical consecutive points, freeze scaling limits on
        outliers, take one SGD step and update the EWMA statistics.

        Args:
            x: Observation as a feature dict; keys are fixed from the
                first sample.

        Returns:
            self.
        """
        self._initialize(x)
        v = self._vector(x)
        if self._phase == "limits":
            self._calibrate_limits(v)
        elif self._phase == "train":
            self._calibrate_train(v)
        else:
            self._detect_step(v)
        return self

    def score_one(self, x: dict) -> float:
        """Return the normalised L1 reconstruction cost of ``x``.

        The raw cost ``r`` (Eq. 11) is divided by the input dimension
        ``d`` so that points reconstructed inside the scaling limits
        score in ``[0, 1]``; outliers beyond the frozen limits may
        exceed 1. The model state is not modified.

        Args:
            x: Observation as a feature dict.

        Returns:
            Normalised reconstruction cost ``r / d``.
        """
        self._initialize(x)
        v_scaled = self._scale(self._vector(x))
        _, z = self._forward(v_scaled)
        return self._cost(v_scaled, z) / len(v_scaled)

    def predict_one(self, x: dict) -> bool:
        """Return whether ``x`` is an outlier under the EWMA threshold.

        Returns ``False`` while the scorer is still self-calibrating
        (Algorithm 1). The model state is not modified.

        Args:
            x: Observation as a feature dict.

        Returns:
            True iff ``r > mu + k * sigma`` (Eq. 16).
        """
        self._initialize(x)
        if not self.calibrated:
            return False
        v_scaled = self._scale(self._vector(x))
        _, z = self._forward(v_scaled)
        return self._is_outlier(self._cost(v_scaled, z))
