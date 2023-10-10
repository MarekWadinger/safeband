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
    ...     p = p.update(x)
    >>> p.mv_conditional(X.iloc[0].values, 0, p.mu, p.var)
    (array([0.61329773]), array([[0.07246737]]), array([0.26919764]))

    TODO: for some reason, first value changed from 0.51220852 to 0.61329773
    in commit 4bb519bcb6312c8761bd25074c5e39d88e195fc4

    >>> p.mv_conditional([0.], 0, np.array([0.]), np.array([[1.]]))
    (array([0.]), array([[1.]]), array([1.]))
    """

    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # TODO: allow any iterable
    def mv_conditional(
            self,
            observed_values: np.ndarray,
            var_idx: int,
            mean: np.ndarray,
            covariance: np.ndarray):
        if isinstance(mean, dict):
            mean = np.array([*mean.values()])
        if isinstance(covariance, pd.DataFrame):
            covariance = covariance.values
        var_idx_: list[int] = [var_idx]
        if len(mean) == 1:  # Univariate case
            conditional_mean = mean
            conditional_covariance = covariance
            conditional_std = np.sqrt(np.diag(conditional_covariance))
        else:  # Multivariate case
            obs_idxs = [i for i in range(len(mean)) if i not in var_idx_]
            if len(observed_values) == len(mean):
                observed_values = np.take(observed_values, obs_idxs)

            cov_XY = covariance[np.ix_(obs_idxs, obs_idxs)]
            cov_XZ = covariance[np.ix_(obs_idxs, var_idx_)]
            cov_ZZ = covariance[np.ix_(var_idx_, var_idx_)]

            regression_coefficients = np.dot(cov_XZ.T, np.linalg.pinv(cov_XY))
            conditional_mean = (
                mean[var_idx_] + np.dot(
                    regression_coefficients,
                    (observed_values - mean[obs_idxs])
                    )
                )
            conditional_covariance = (
                cov_ZZ - np.dot(regression_coefficients, cov_XZ)
                )
            # TODO: handle very small negative covariance using tolerance
            conditional_std = np.sqrt(np.diag(conditional_covariance))
        return conditional_mean, conditional_covariance, conditional_std


if __name__ == '__main__':
    import doctest
    doctest.testmod()
