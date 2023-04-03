
from ompl import base as ob
from ompl import geometric as og
import numpy as np
import mujoco
import copy

from air_hockey_challenge.constraints.constraints import Constraint
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian, link_to_xml_name

class AirHockeyOmplTableConstraint(ob.Constraint):
    def __init__(self, ambient_dim: int,  airhockey_constraint: Constraint):
        self.__codim = airhockey_constraint.output_dim
        super(AirHockeyOmplTableConstraint, self).__init__(ambient_dim, self.__codim)
        self.__constraint = airhockey_constraint

    def function(self, q, out):
        out[:,:] = self.__constraint.fun(q, q*0).copy()

    def jacobian(self, q, out):
        out[:,:] = self.__constraint.jacobian(q, q*0).copy()

class AirHockeyCircleConstraint(ob.Constraint):
    def __init__(self, 
        n_joints: int,
        robot_model: mujoco.MjModel,
        robot_data: mujoco.MjData,
        circle_center: np.ndarray,
        circle_radius: float
    ):
        self.__codim = 1
        self.__ambient_dim = n_joints
        super(AirHockeyCircleConstraint, self).__init__(self.__ambient_dim, self.__codim)
        self.robot_model = copy.deepcopy(robot_model)
        self.robot_data= copy.deepcopy(robot_data)
        self.__circle_center = circle_center
        self.__circle_radius = circle_radius

    def function(self, q, out):
        pose = forward_kinematics(self.robot_model, self.robot_data, q)[0]
        out[0] = np.linalg.norm(self.__circle_center - pose)-self.__circle_radius

    def jacobian(self, q, out):
        pose = forward_kinematics(self.robot_model, self.robot_data, q)[0]
        robot_jac = jacobian(self.robot_model, self.robot_data, q)[:3, :self.__ambient_dim]
        nrm = np.linalg.norm(pose)
        # print(np.linalg.norm(pose))
        cartesian_constr_jac = np.array([[1, 0, 0]], dtype=float)
        if np.isfinite(nrm) and nrm > 0:
            cartesian_constr_jac[0, :] = pose / nrm
        # print((cartesian_constr_jac @ robot_jac)[0])
        out[:] = (cartesian_constr_jac @ robot_jac).flatten()