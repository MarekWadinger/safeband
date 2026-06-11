from typing import cast

import numpy as np
import pandas as pd
from river import proba


class MultivariateGaussian(proba.MultivariateGaussian):
    """Multivariate normal distribution with parameters mu and var.

    Parameters
    ----------
    seed
        Random number generator seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd

    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.random((8, 3)),
    ...                  columns=["red", "green", "blue"])

    >>> p = MultivariateGaussian()
    >>> p.n_samples
    0.0
    >>> for x in X.to_dict(orient="records"):
    ...     p.update(x)
    >>> p.mu
    {'blue': 0.517..., 'green': 0.386..., 'red': 0.415...}
    >>> p.mv_conditional(p.mu, "red", p.mu, p.var)
    (array([0.415...]), array([[0.053...]]), array([0.232...]))
    >>> p.mv_conditional(x, "red", p.mu, p.var)
    (array([0.461...]), array([[0.053...]]), array([0.232...]))

    >>> p.mv_conditional([0., 0.], 0, np.array([0., 1.]),
    ...                  np.array([[1., 0], [0.5, 1.]]))
    (array([-0.5]), array([[0.75]]), array([0.866...]))

    >>> p.mv_conditional(list(x.values()), 0, p.mu, p.var)
    Traceback (most recent call last):
    ...
    ValueError: Arguments must be either dict, str, dict, pd.DataFrame or
    np.ndarray, int, np.ndarray, np.ndarray.

    """

    def __init__(self, seed=None) -> None:
        super().__init__(seed=seed)

    def mv_conditional(
        self,
        observed_values: dict[str, float] | np.ndarray,
        var_idx: str | int,
        mean: dict[str, float] | np.ndarray,
        covariance: pd.DataFrame | np.ndarray,
    ):
        if (
            isinstance(observed_values, dict)
            and isinstance(mean, dict)
            and isinstance(var_idx, str)
            and isinstance(covariance, pd.DataFrame)
        ):
            _ov = cast("dict[str, float]", observed_values)
            _mn = cast("dict[str, float]", mean)
            observed_values = np.array([_ov[key] for key in _mn])
            var_idx = list(mean.keys()).index(var_idx)
            mean = np.array([*mean.values()])
            covariance = covariance.to_numpy()
        elif (
            isinstance(observed_values, (list, np.ndarray))
            and isinstance(mean, np.ndarray)
            and isinstance(var_idx, int)
            and isinstance(covariance, np.ndarray)
        ):
            pass
        else:
            msg = (
                "Arguments must be either dict, str, dict, pd.DataFrame or "
                "np.ndarray, int, np.ndarray, np.ndarray."
            )
            raise ValueError(
                msg,
            )
        # After both branches, mean and observed_values are always np.ndarray
        mean = cast("np.ndarray", mean)
        observed_values = cast("np.ndarray", observed_values)
        var_idx_: list[int] = [var_idx]  # type: ignore[list-item]
        if len(mean) == 1:  # Univariate case
            conditional_mean = mean
            conditional_covariance = covariance
            conditional_std = np.sqrt(np.diag(conditional_covariance))
        else:  # Multivariate case
            obs_idxs = [i for i in range(len(mean)) if i not in var_idx_]
            if len(observed_values) == len(mean):
                observed_values = np.take(observed_values, obs_idxs)

            cov_XY = np.nan_to_num(covariance[np.ix_(obs_idxs, obs_idxs)])
            cov_XZ = covariance[np.ix_(obs_idxs, var_idx_)]
            cov_ZZ = covariance[np.ix_(var_idx_, var_idx_)]

            regression_coefficients = np.dot(cov_XZ.T, np.linalg.pinv(cov_XY))
            conditional_mean = mean[var_idx_] + np.dot(
                regression_coefficients,
                (observed_values - mean[obs_idxs]),
            )
            conditional_covariance = cov_ZZ - np.dot(
                regression_coefficients,
                cov_XZ,
            )

            conditional_covariance[conditional_covariance < 0] = 1e-8
            conditional_std = np.sqrt(np.diag(conditional_covariance))
        return conditional_mean, conditional_covariance, conditional_std


if __name__ == "__main__":
    import doctest

    doctest.testmod()
