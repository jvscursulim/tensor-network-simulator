import numpy as np

from tsim.gates import Gate


class State(Gate):
    def __init__(self):
        self.params = None


class Zero(State):
    @classmethod
    def _matrix(self):
        return np.array([1.0, 0.0])


class One(State):
    @classmethod
    def _matrix(self):
        return np.array([0.0, 1.0])
