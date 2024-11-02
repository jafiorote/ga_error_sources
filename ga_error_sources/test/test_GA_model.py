import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from ga_error_sources.model.gaModel import GAModel


class TestGAModel(unittest.TestCase):
    
    def setUp(self):
        self.model = GAModel()

    def test_get_transitions_matrix_basic(self):
        arr_prob = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.2, 0.1, 0.7]])
        transitions, idxs = self.model.get_transitions_matrix(arr_prob)
        self.assertEqual(transitions.shape, (9, 9))  
        self.assertEqual(len(idxs), 9)

    def test_get_best_path_maximization(self):
        state_probs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        pathway, step_probs = self.model.get_best_path(state_probs)

        expected_pathway = [[0, 0], [1, 1], [2, 2]]
        expected_step_probs = [0.1, 0.5 / (0.2 + 0.5), 0.9 / (0.3 + 0.6 + 0.9)]

        self.assertEqual(pathway, expected_pathway)
        assert_array_almost_equal(step_probs, expected_step_probs)

    def test_get_best_path_minimization(self):
        state_probs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        pathway, step_probs = self.model.get_best_path(state_probs, minimize=True)

        expected_pathway = [[2, 0], [2, 1], [2, 2]]
        expected_step_probs = [0.9, 0.8 / (0.5 + 0.8), 0.7 / (0.4 + 0.7) ]

        self.assertEqual(pathway, expected_pathway)
        assert_array_almost_equal(step_probs, expected_step_probs)

    def test_get_best_path_custom_init_state(self):
        state_probs = np.array([[0.2, 0.3], [0.4, 0.8]])
        init_state = [1, 0]
        pathway, step_probs = self.model.get_best_path(state_probs, init_state=init_state)

        expected_pathway = [[1, 0], [1, 1]]
        expected_step_probs = [0.4, 0.8 / (0.8 + 0.3)]

        self.assertEqual(pathway, expected_pathway)
        assert_array_almost_equal(step_probs, expected_step_probs)

    def test_get_transitions_matrix_empty_input(self):
        arr_prob = np.zeros((3, 3))
        transitions, idxs = self.model.get_transitions_matrix(arr_prob)
        assert_array_almost_equal(transitions, np.zeros((9, 9)))

if __name__ == '__main__':
    unittest.main()
