import warnings

import numpy as np
from scipy.stats import multivariate_normal

from river import covariance, proba

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
    >>> X
            red     green      blue
    0  0.374540  0.950714  0.731994
    1  0.598658  0.156019  0.155995
    2  0.058084  0.866176  0.601115
    3  0.708073  0.020584  0.969910
    4  0.832443  0.212339  0.181825
    5  0.183405  0.304242  0.524756
    6  0.431945  0.291229  0.611853
    7  0.139494  0.292145  0.366362

    >>> p = MultivariateGaussian()
    >>> p.n_samples
    0.0
    >>> for x in X.to_dict(orient="records"):
    ...     p = p.update(x)
    >>> p.mv_conditional(X.iloc[0].values, 0, np.array(p.mu), p.var)
    (array([0.51220852]), array([[0.07246737]]), array([0.26919764]))
    
    >>> p.mv_conditional([0.], 0, np.array([0.]), np.array([[1.]]))
    (array([0.]), array([[1.]]), array([1.]))
    """

    def __init__(self, seed=None):
        super().__init__(seed=seed)
        

    # TODO: allow any iterable
    def mv_conditional(
            self,
            observed_values: np.array,
            var_idx: int,
            mean: np.array,
            covariance: np.array):
        var_idx = [var_idx]
        if len(mean) == 1:  # Univariate case
            conditional_mean = mean
            conditional_covariance = covariance
            conditional_std = np.sqrt(np.diag(conditional_covariance))
        else:  # Multivariate case
            obs_idxs = [i for i in range(len(mean)) if i not in var_idx]
            if len(observed_values) == len(mean):
                observed_values = np.take(observed_values, obs_idxs)

            cov_XY = covariance[np.ix_(obs_idxs, obs_idxs)]
            cov_XZ = covariance[np.ix_(obs_idxs, var_idx)]
            cov_ZZ = covariance[np.ix_(var_idx, var_idx)]

            regression_coefficients = np.dot(cov_XZ.T, np.linalg.pinv(cov_XY))
            conditional_mean = (
                mean[var_idx] + np.dot(
                    regression_coefficients,
                    (observed_values - mean[obs_idxs])
                    )
                )
            conditional_covariance = (
                cov_ZZ - np.dot(regression_coefficients, cov_XZ)
                )
            conditional_std = np.sqrt(np.diag(conditional_covariance))
        return conditional_mean, conditional_covariance, conditional_std
    

if __name__ == '__main__':
    import doctest
    doctest.testmod()