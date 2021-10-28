import unittest
import numpy as np
import tensornetwork as tn
from tsim.states import Zero, One


class TestZero(unittest.TestCase):
    def setUp(self) -> None:
        self.state = Zero()
        self.expected_matrix = np.array([1.0, 0.0])

    def test_zero_matrix(self):

        self.assertTrue(np.allclose(self.state.matrix, self.expected_matrix))

    def test_zero_tensor(self):

        self.assertTrue(np.allclose(self.state.node.tensor, self.expected_matrix))


class TestOne(unittest.TestCase):
    def setUp(self) -> None:
        self.state = One()
        self.expected_matrix = np.array([0.0, 1.0])

    def test_one_matrix(self):

        self.assertTrue(np.allclose(self.state.matrix, self.expected_matrix))

    def test_one_tensor(self):

        self.assertTrue(np.allclose(self.state.node.tensor, self.expected_matrix))
