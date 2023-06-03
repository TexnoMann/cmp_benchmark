from __future__ import print_function
import numpy as np
import sys, os
from abc import ABC,abstractmethod

from ompl import util as ou
from ompl import base as ob

from itmobotics_sim.utils.robot import EEState

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def list2vec(l):
    ret = ou.vectorDouble()
    for e in l:
        ret.append(e)
    return ret

def ompl2numpy(ompl, ret: np.ndarray):
    for i in range(0, ret.shape[0]):
        ret[i] = ompl[i]

def numpy2ompl(arr, ret):
    for i in range(0, arr.shape[0]):
        ret[i] = arr[i]

class BenchmarkConstrainedScene(ABC):

    def is_state_valid(self, state) -> bool:
        q = np.zeros(self.state_space.getDimension())
        ompl2numpy(state, q)
        return self.is_q_valid(q)

    @abstractmethod
    def is_q_valid(self, q) -> bool:
        raise RuntimeError("Please realize is_q_valid() method")
    
    @abstractmethod
    def get_constrained_configuration_from_workspace( self, tf_robot2: EEState, initial_q: np.ndarray):
        pass

    @abstractmethod
    def get_workspace_from_configuration( self, initial_q: np.ndarray):
        pass

    @property
    @abstractmethod
    def constraint(self)->ob.Constraint:
        pass

    @property
    @abstractmethod
    def state_space(self)->ob.StateSpace:
        pass
    
class NewtonRaphsonProjectionEvaluator(ob.ProjectionEvaluator):
    def __init__(self, constr_space):
        super(NewtonRaphsonProjectionEvaluator, self).__init__(constr_space)
        self.__constr_space = constr_space
        self.__constraint = constr_space.getConstraint()

    def getDimension(self):
        return self.__constr_space.getDimension()

    def defaultCellSizes(self):
        self.cellSizes_ = list2vec([0.1 for _ in range(0, self.__constraint.getCoDimension())])

    def project(self, state, projection):
        print("Project")
        i = 0
        q = np.zeros(self.__constraint.getAmbientDimension())
        ompl2numpy(state, q)
        x = np.zeros(self.__constraint.getCoDimension())
        J = np.zeros((self.__constraint.getCoDimension(), self.__constraint.getAmbientDimension()))
        self.__constraint.function(q, x)
        # print(self.getTolerance())
        while x.dot(x) >= self.__constraint.getTolerance()**2:
            if i > self.__constraint.getMaxIterations():
                return False
            self.__constraint.jacobian(q, J)
            q = q - np.linalg.pinv(J)@x
            self.__constraint.function(q, x)
            i+=1
        numpy2ompl(q, projection)
        self.__constraint.function(q, x)
        return True