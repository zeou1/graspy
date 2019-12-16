import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from sklearn.exceptions import NotFittedError

from graspy.cluster.pgmm import PartitionalGaussianCluster
from graspy.embed.ase import AdjacencySpectralEmbed
from graspy.simulations.simulations import sbm


def test_inputs():
    # Generate random data
    X = np.random.normal(0, 1, size=(20, 3))
    # max_components < min_components
    with pytest.raises(ValueError):
        pclust = PartitionalGaussianCluster(max_components=0)

    # max_components integer
    with pytest.raises(TypeError):
        pclust = PartitionalGaussianCluster(min_components=1, max_components="1")

    # covariance type is not an array, string or list
    with pytest.raises(TypeError):
        pclust = PartitionalGaussianCluster(covariance_type=1)

    # covariance type is not in ['spherical', 'diag', 'tied', 'full']
    with pytest.raises(ValueError):
        pclust = PartitionalGaussianCluster(covariance_type="graspy")

    # max_cluster > n_samples when max_cluster is not None
    with pytest.raises(ValueError):
        pclust = PartitionalGaussianCluster(1001)
        pclust.fit(X)


def test_predict_without_fit():
    # Generate random data
    X = np.random.normal(0, 1, size=(20, 3))

    with pytest.raises(NotFittedError):
        pclust = PartitionalGaussianCluster()
        pclust.predict(X)


def test_no_y():
    np.random.seed(2)

    n = 20
    d = 3

    X1 = np.random.normal(2, 0.5, size=(n, d))
    X2 = np.random.normal(-2, 0.5, size=(n, d))
    X = np.vstack((X1, X2))

    pclust = PartitionalGaussianCluster(n_init=2)
    pclust.fit(X)

    inrange = (pclust.n_components_ >= 1) and (pclust.n_components_ <= n*2)

    assert_equal(inrange, True)

