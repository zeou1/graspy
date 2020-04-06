# Copyright 2020 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

class SeedlessProcrustes():
    r"""

    Matches datasets by iterating over a decreasing sequence of
    lambdas, the penalization parameters.
    If lambda is big, the function is more concave, so we iterate,
    starting from Lambda = .5, and decreasing each time by alpha.
    This method takes longer, but is more likely to converge to 
    the true solution, since we start from a more concave problem 
    and iteratively solve it by setting
    lambda = alpha * lambda, for alpha \in (0,1).
    
    Parameters
    ----------
        lambda_init : float, optional (default 0.5)
            the initial value of lambda for penalization

        lambda_final : float, optional (deafault: 0.001)
            for termination

        alpha : float, optional (default 0.95)
            the parameter for which lambda is multiplied by

        optimal_transport_eps : float, optional
            the tolerance for the optimal transport problem

        iteration_eps : float, optional
            the tolerance for the iterative problem

        num_reps : int, optional
            the number of reps for each subiteration
            
    Attributes
    ----------
        P : array, size (n, m) where n and md are the sizes of two datasets 
            final matrix of optimal transports
            
        Q : array, size (d, d) where d is the dimensionality of the datasets
            final orthogonal matrix
    
    References
    ----------
    .. [1] Agterberg, J.
    """
    
    def __init__(
        self, lambda_init=0.5, lambda_final=0.001, alpha=0.95,
        optimal_transport_eps=0.01, iterative_eps=0.01, num_reps=100):
        if type(lambda_init) is not float:
            raise TypeError()
        if type(lambda_final) is not float:
            raise TypeError()
        if type(alpha) is not float:
            raise TypeError()
        if type(optimal_transport_eps) is not float:
            raise TypeError()
        if type(iterative_eps) is not float:
            raise TypeError()
        if type(num_reps) is not int:
            raise TypeError()
        if alpha < 0 or alpha > 1:
            raise ValueError(
                "{} is an invalid value of alpha must be strictly between 0 and 1".format(
                alpha)
            )
        if optimal_transport_eps <= 0:
            raise ValueError(
                "{} is an invalud value of the optimal transport eps, must be postitive".format(
                    optimal_transport_eps
                )
            )
        if iterative_eps <= 0:
            raise ValueError(
                "{} is an invalud value of the iterative eps, must be postitive".format(
                    iterative_eps
                )
            )
        if num_reps < 1:
            raise ValueError(
                "{} is invalid number of repetitions, must be greater than 1".format(
                    num_reps
                )
            )
        self.lambda_init = lambda_init
        self.lambda_final = lambda_final
        self.alpha = alpha
        self.optimal_transport_eps = optimal_transport_eps
        self.iterative_eps = iterative_eps
        self.num_reps = num_reps
    
    def _procrustes(self, X, Y, P_i):
        u, w, vt = np.linalg.svd(X.T @ P_i @ Y)
        Q = u.dot(vt)
        return Q

    def _optimal_transport(self, X, Y, Q, lambd=0.1):
        n, d = X.shape
        m, _ = Y.shape

        X = X @ Q
        C = np.linalg.norm(X.reshape(n, 1, d) - Y.reshape(1, m, d), axis=2) ** 2

        r = 1 / n
        c = 1 / m
        P = np.exp(-lambd * C)
        u = np.zeros(n)
        while np.max(np.abs(u - np.sum(P, axis=1))) > self.optimal_transport_eps:
            u = np.sum(P, axis=1)
            P = r * P / u.reshape(-1, 1)
            v = np.sum(P, axis=0)
            P = c * (P.T / v.reshape(-1, 1)).T
        return P

    def _iterative_optimal_transport(self, X, Y, Q=None, lambd=0.01):
        _, d = X.shape
        if Q is None:
            Q = np.eye(d)

        for i in range(self.num_reps):
            P_i = self._optimal_transport(X, Y, Q)
            Q = self._procrustes(X, Y, P_i)
            c = np.linalg.norm(X @ Q - P_i @ Y, ord='fro')
            if c < self.iterative_eps:
                break
        return P_i, Q

    def _sign_flips(self, X, Y):
        X_medians = np.median(X, axis=0)
        Y_medians = np.median(Y, axis=0)
        val = np.multiply(X_medians, Y_medians)
        t = (val > 0) * 2 - 1
        return np.diag(t)
    
    def fit(self, X, Y, Q=None, P=None):
        '''
        matches the datasets

        Parameters
        ----------
        X: np.ndarray, shape (n, d)
            first dataset of vectors

        Y: np.ndarray, shape (m, d)
            second dataset of vectors

        Q: np.ndarray, shape (d, d) or None, optional (default=None)
            an initial guess for the othogonal alignment matrix, if such exists.
            If None - initializes using an initial guess for P. If that is also
            None - initializes using the median heuristic sign flips.

        P: np.ndarray, shape (n, m) or None, optional (default=None)
            an initial guess for the initial transpot matrix.
            Only matters if Q=None. 

        Returns
        -------
        self: returns an instance of self
        '''
        if Q is None:
            if P is None:
                Q = self._sign_flips(X, Y)
            else:
                Q = self._procrustes(X, Y, P)

        lambda_current = self.lambda_init
        while lambda_current > self.lambda_final:
            P_i, Q = self._iterative_optimal_transport(X, Y, Q,
                                                      lambd=lambda_current)
            lambda_current = self.alpha * lambda_current
        self.Q = Q
        self.P = P_i
        return self

    def fit_predict(self, X, Y, Q=None):
        '''
        matches datasets, returning the final orthogonal alignment solution

        Parameters
        ----------
        X: np.ndarray, shape (n, d)
            first dataset of vectors

        Y: np.ndarray, shape (m, d)
            second dataset of vectors

        Q: np.ndarray, shape (d, d) or None, optional (default=None)
            an initial guess for the othogonal alignment matrix, if such exists.
            If None - initializes using an initial guess for P. If that is also
            None - initializes using the median heuristic sign flips.

        P: np.ndarray, shape (n, m) or None, optional (default=None)
            an initial guess for the initial transpot matrix.
            Only matters if Q=None. 

        Returns
        -------
        Q : array, size (d, d) where d is the dimensionality of the datasets
            final orthogonal matrix
        '''
        self.fit(X, Y, Q)
        return self.Q
