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
