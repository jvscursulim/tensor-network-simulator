import unittest
import numpy as np
import tensornetwork as tn

from tsim.gates import Gate


class TestGate(unittest.TestCase):
    def test_gate_notimplemented(self):
        with self.assertRaises(NotImplementedError):
            Gate().matrix
