import unittest
import numpy as np
import tensornetwork as tn

from tsim.gates import Gate, RZ, RY, RX, CX


class TestGate(unittest.TestCase):
    def test_gate_notimplemented(self):
        with self.assertRaises(NotImplementedError):
            Gate().matrix


class TestRZ(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = RZ(1.5)
        self.expected_matrix = np.array(
            [[np.exp(-1j * 0.75), 0.0], [0.0, np.exp(1j * 0.75)]]
        )

    def test_RX_matrix(self):

        self.assertTrue(np.allclose(self.gate.matrix, self.expected_matrix))

    def test_RX_tensor(self):

        self.assertTrue(np.allclose(self.gate.node.tensor, self.expected_matrix))


class TestRY(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = RY(1.5)
        self.expected_matrix = np.array(
            [
                [np.cos(0.75), -np.sin(0.75)],
                [np.sin(0.75), np.cos(0.75)],
            ]
        )

    def test_RY_matrix(self):

        self.assertTrue(np.allclose(self.gate.matrix, self.expected_matrix))

    def test_RY_tensor(self):

        self.assertTrue(np.allclose(self.gate.node.tensor, self.expected_matrix))


class TestRX(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = RX(1.5)
        self.expected_matrix = np.array(
            [
                [np.cos(0.75), -1.0j * np.sin(0.75)],
                [-1.0j * np.sin(0.75), np.cos(0.75)],
            ]
        )

    def test_RX_matrix(self):

        self.assertTrue(np.allclose(self.gate.matrix, self.expected_matrix))

    def test_RX_tensor(self):

        self.assertTrue(np.allclose(self.gate.node.tensor, self.expected_matrix))


class TestCX(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = CX()
        self.expected_matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        ).reshape(2, 2, 2, 2)

    def test_CX_matrix(self):

        self.assertTrue(np.allclose(self.gate.matrix, self.expected_matrix))

    def test_CX_tensor(self):

        self.assertTrue(np.allclose(self.gate.node.tensor, self.expected_matrix))
