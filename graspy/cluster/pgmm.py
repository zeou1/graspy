# Copyright 2019 NeuroData (http://neurodata.io)
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
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid

from .base import BaseCluster
from .gclust import GaussianCluster


class PartitionalGaussianCluster(BaseCluster):
    r"""
    Gaussian Mixture Model (GMM)

    Partitional Gaussian Mixture Modeling (GMM) Clustering
    Algorithm. Given a set of data, this algorithm
    fits GMMs with different cluster numbers and covariance
    constraints and selects the model that has the best
    Bayesian Information Criterion (BIC). Repeats this 
    process until the single cluster models have the
    best BICs.

    Parameters
    ----------
    max_components : int, default=2.
        The maximum number of mixture components to consider. Must be greater
        than or equal to 2.

    covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}, optional
        String or list/array describing the type of covariance parameters to use.
        If a string, it must be one of:

        - 'full'
            each component has its own general covariance matrix
        - 'tied'
            all components share the same general covariance matrix
        - 'diag'
            each component has its own diagonal covariance matrix
        - 'spherical'
            each component has its own single variance
        - 'all'
            considers all covariance structures in ['spherical', 'diag', 'tied', 'full']
        If a list/array, it must be a list/array of strings containing only
            'spherical', 'tied', 'diag', and/or 'spherical'.

        tol : float, defaults to 1e-3.
            The convergence threshold. EM iterations will stop when the
            lower bound average gain is below this threshold.

        reg_covar : float, defaults to 1e-6.
            Non-negative regularization added to the diagonal of covariance.
            Allows to assure that the covariance matrices are all positive.

        max_iter : int, defaults to 100.
            The number of EM iterations to perform.

        n_init : int, defaults to 1.
            The number of initializations to perform. The best results are kept.

        init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
            The method used to initialize the weights, the means and the
            precisions.
            Must be one of::
                'kmeans' : responsibilities are initialized using kmeans.
                'random' : responsibilities are initialized randomly.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by ``np.random``.


    Attributes
    ----------
    n_components_ : int
        Optimal number of components based on BIC.
    """

    def __init__(
        self,
        max_components=2,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        random_state=None,
    ):

        if isinstance(max_components, int):
            if max_components <= 0:
                msg = "max_components must be >= 1 or None."
                raise ValueError(msg)
        elif max_components is not None:
            msg = "max_components must be an integer or None, not {}.".format(
                type(max_components)
            )
            raise TypeError(msg)

        if isinstance(covariance_type, (np.ndarray, list)):
            covariance_type = np.unique(covariance_type)
        elif isinstance(covariance_type, str):
            if covariance_type == "all":
                covariance_type = ["spherical", "diag", "tied", "full"]
            else:
                covariance_type = [covariance_type]
        else:
            msg = "covariance_type must be a numpy array, a list, or "
            msg += "string, not {}".format(type(covariance_type))
            raise TypeError(msg)

        for cov in covariance_type:
            if cov not in ["spherical", "diag", "tied", "full"]:
                msg = (
                    "covariance structure must be one of "
                    + '["spherical", "diag", "tied", "full"]'
                )
                msg += " not {}".format(cov)
                raise ValueError(msg)

        new_covariance_type = []
        for cov in ["spherical", "diag", "tied", "full"]:
            if cov in covariance_type:
                new_covariance_type.append(cov)

        self.max_components = max_components
        self.covariance_type = new_covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Performs partitional GMM. The algorithms
        relies on two arrays, y and complete. y 
        is a matrix that indicates the partitions
        at each level of the hierarchy. complete
        keeps track of which data points have
        been partitioned into single component
        models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------

        y : array-like, shape (n_samples,n_iters), optional (default=None)
            List of labels for X if available. Each column
            indicates a partition at a particular level. Note
            that the unique clusters are identified by the 
            whole row, not just the last column
        """

        # Deal with number of clusters
        if self.max_components > X.shape[0]:
            msg = "max_components must be >= n_samples, but max_components = "
            msg += "{}, n_samples = {}".format(self.max_components, X.shape[0])
            raise ValueError(msg)

        n_samples = X.shape[0]
        # first column is all 0s (all data points together)
        y = np.zeros([n_samples, 1])
        complete = np.zeros([n_samples, 1])

        default_params = {
            "max_components": self.max_components,
            "covariance_type": self.covariance_type,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "init_params": self.init_params,
            "random_state": self.random_state,
        }

        counter = 0
        while not np.all(complete == 1):
            counter = counter + 1
            print("Partition level: " + str(counter))

            # unique partition "histories"
            # e.g. [0,1,0] and [0,0,0] were partitioned
            # in the first iteration
            clusters = np.unique(y, axis=0)
            y = np.concatenate((y, np.zeros([n_samples, 1])), axis=1)

            for cluster in clusters:
                # get all data points in the same partition/leaf
                cluster_idxs = [(y[i, :-1] == cluster).all() for i in range(n_samples)]
                # remove points that are 'complete'
                cluster_idxs = cluster_idxs & (complete[:, -1] == 0)

                # all of these indices have finished
                if np.sum(cluster_idxs) == 0:
                    continue

                # data points in this cluster
                X_c = X[cluster_idxs, :]

                # may need to modify max number of clusters
                # to the number of data points
                # TODO: change max components to be limited according to dimension etc
                if default_params["max_components"] > np.sum(cluster_idxs):
                    params = default_params
                    params["max_components"] = int(np.sum(cluster_idxs))
                else:
                    params = default_params

                gclust = GaussianCluster(**params)

                labels = gclust.fit_predict(X_c)

                y[cluster_idxs, -1] = labels

                [_, unq_idxs, unq_counts] = np.unique(
                    labels, return_index=True, return_counts=True
                )

                # for points that were assigned their own cluster
                # mark them as complete
                for l, count in enumerate(unq_counts):
                    if count == 1:
                        orig_idxs = np.argwhere(cluster_idxs)
                        complete[orig_idxs[unq_idxs[l]], -1] = 1

                # if BIC said that one component was best
                # mark all points as complete
                if gclust.n_components_ == 1:
                    complete[cluster_idxs, -1] = 1

        self.n_components_ = len(np.unique(y, axis=1))

        return y
