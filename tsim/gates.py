from typing import Union
import tensornetwork as tn
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, List


class Gate(ABC):
    @classmethod
    def _matrix(self) -> np.array:
        raise NotImplementedError

    @property
    def matrix(self) -> np.array:
        return self._matrix()

    @property
    def node(self) -> tn.Node:
        return tn.Node(self._matrix())

    def __init__(self, *params):
        self.params = list(params)


class RZ(Gate):
    """RZ(theta)
    Single qubit Z rotation.

    """

    def _matrix(self):
        theta = self.params[0]
        return np.array([[np.exp(-1j * theta / 2), 0.0], [0.0, np.exp(1j * theta / 2)]])


class RY(Gate):
    """RY(theta)
    Single qubit Y rotation.

    """

    def _matrix(self):
        theta = self.params[0]
        return np.array(
            [
                [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                [np.sin(theta / 2.0), np.cos(theta / 2.0)],
            ]
        )


class RX(Gate):
    """RX(theta)
    Single qubit X rotation.

    """

    def _matrix(self):
        theta = self.params[0]
        return np.array(
            [
                [np.cos(theta / 2.0), -1.0j * np.sin(theta / 2.0)],
                [-1.0j * np.sin(theta / 2.0), np.cos(theta / 2.0)],
            ]
        )


class CX(Gate):
    """CX
    Two-qubit CNOT gate.

    """

    def _matrix(self):

        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        ).reshape(2, 2, 2, 2)
