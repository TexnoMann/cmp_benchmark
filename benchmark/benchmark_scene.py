from __future__ import print_function
import numpy as np
import sys, os
from abc import ABC,abstractmethod

from ompl import base as ob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BenchmarkConstrainedScene(ABC):
    @abstractmethod
    def is_state_valid(state) -> bool:
        pass

    @abstractmethod
    @property
    def constraint(state)->ob.Constraint:
        pass

    @abstractmethod
    @property
    def state_space(self)->ob.StateSpace:
        pass