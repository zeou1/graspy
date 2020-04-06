# Anton Alyakin
# aalyaki1 [at] jhu.edu
# 04.06.2020

import unittest

import numpy as np
from scipy import stats
from graspy.align import SeedlessProcrustes

class TestSeedlessProcrustes(unittest.TestCase):
    def test_bad_kwargs(self):
        with self.asertRaises(TypeError):
            SeedlessProcrustes(lambda_init="oops")
        with self.asertRaises(TypeError):
            SeedlessProcrustes(lambda_final="oops")
        with self.asertRaises(TypeError):
            SeedlessProcrustes(alpha="oops")
        with self.asertRaises(TypeError):
            SeedlessProcrustes(optimal_transport_eps="oops")
        with self.asertRaises(TypeError):
            SeedlessProcrustes(iterative_eps="oops")
        with self.asertRaises(TypeError):
            SeedlessProcrustes(num_reps=3.15)
        with self.asertRaises(ValueError):
            SeedlessProcrustes(alpha=-1)
        with self.asertRaises(ValueError):
            SeedlessProcrustes(optimal_transport_eps=-1)
        with self.asertRaises(ValueError):
            SeedlessProcrustes(iterative_eps=-1)
        with self.asertRaises(ValueError):
            SeedlessProcrustes(num_reps=0)

    def test_sign_flips(self):
        X = np.arange(6).reshape(3, 2)
        Y = np.arange(6).reshape(3, 2) @ np.diag([1, -1]) + 0.5
        test = _sign_flips(X, Y) 
        answer = np.array([[1, 0], [0, -1]])
        self.assertTrue(np.all(test == answer))

    def test_iterative_optimal_transport(self):
        np.random.seed(314)
        X = np.random.normal(1, 0.2, 1000).reshape(250, 4)
        Y = np.concatenate([X, X])
        W = stats.ortho_group.rvs(4)
        Y = Y @ W
        ​
        aligner = SeedlessProcrustes()
        _, Q = aligner._iterative_optimal_transport(X, Y)
        self.assertTrue(np.all(np.isclose(Q, W)))

    def test_matching_datasets(self):
        np.random.seed(31412)
        X = np.random.normal(0.2, 0.02, 1000).reshape(200, 5)
        Y = np.concatenate([X, X])
        W = stats.ortho_group.rvs(4)
        Y = Y @ W
        ​
        aligner = SeedlessProcrustes()
        Q = aligner.fit_transform(X, Y)
        self.assertTrue(np.all(np.isclose(Q, W)))

if __name__ == "__main__":
    unittest.main()
