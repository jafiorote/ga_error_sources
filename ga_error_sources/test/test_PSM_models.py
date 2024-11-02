import numpy as np
import unittest
from scipy.stats import poisson, norm

from ga_error_sources.model.proteinSystemModel import ProteinSystemModel
from ga_error_sources.model.normPSM import NormPSM


class TestProteinSystemModel(unittest.TestCase):
    
    def setUp(self):
        self.model = ProteinSystemModel(M=10, i_0=0.1, i_nat=1.0, sigma2_0=0.05, n_bins=5, verbose=False)

    def test_initialization(self):
        self.assertEqual(self.model.get_M(), 10)
        self.assertEqual(self.model.get_i_0(), 0.1)
        self.assertEqual(self.model.get_i_nat(), 1.0)
        self.assertEqual(self.model.get_sigma2_0(), 0.05)
        self.assertEqual(self.model.get_n_bins(), 5)

    def test_alpha_calculation(self):
        alpha = self.model.get_alpha()
        self.assertAlmostEqual(alpha, 0.9, places=4)

    def test_beta_calculation(self):
        beta = self.model.get_beta(5)
        self.assertAlmostEqual(beta, 0.2014, places=4)

    def test_gama_calculation(self):
        gama = self.model.get_gama(5)
        self.assertAlmostEqual(gama, 0.5, places=4)

    def test_create_data(self):
        data, bins_center = self.model.create_data()
        self.assertEqual(len(data), self.model.get_n_bins() + 1)
        self.assertEqual(len(bins_center), self.model.get_n_bins())

    def test_get_poisson_weights(self):
        weights = self.model.get_poisson_weights()
        expected_weights = np.array([poisson.pmf(n, 1) for n in range(self.model.get_M() + 1)])
        np.testing.assert_almost_equal(weights.flatten(), expected_weights, decimal=4)


class TestNormPSM(unittest.TestCase):

    def setUp(self):
        self.model = NormPSM(M=10, i_0=0.1, i_nat=1.0, sigma2_0=0.05, n_bins=5, verbose=False)

    def test_statistical_func(self):
        prob_density = self.model.statistical_func(0.5, sigma2=0.1, expec=0.5)
        expected_density = norm.pdf(0.5, loc=0.5, scale=np.sqrt(0.1))
        self.assertAlmostEqual(prob_density, expected_density, places=4)

    def test_prob_interval(self):
        prob = self.model.prob_interval(0.4, 0.6, sigma2=0.1, expec=0.5)
        expected_prob = norm.cdf(0.6, loc=0.5, scale=np.sqrt(0.1)) - norm.cdf(0.4, loc=0.5, scale=np.sqrt(0.1))
        self.assertAlmostEqual(prob, expected_prob, places=4)

if __name__ == '__main__':
    unittest.main()
