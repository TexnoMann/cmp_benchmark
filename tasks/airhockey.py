import os, sys

import numpy as np
from ompl import base as ob
from ompl import geometric as og

import os
import sys
import time

import numpy as np
from spatialmath import SE3, SO3
from spatialmath import base as sb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.benchmark_scene import BenchmarkConstrainedScene
from benchmark.benchmark import ompl2numpy, numpy2ompl

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot
from itmobotics_sim.utils.controllers import EEPositionToEEVelocityController, EEVelocityToJointVelocityController, JointTorquesController

def create_tf_for_plane(center: np.ndarray, normal: np.ndarray):
    plate_normal = normal/np.linalg.norm(normal)
    plate_orto_tf_x = np.cross(plate_normal, np.array([1,0,0]))
    plate_orto_tf_y = np.cross(plate_normal, plate_orto_tf_x)
    plate_virtual_tf = SE3(
        np.concatenate(
            [
                np.concatenate([np.atleast_2d(plate_orto_tf_x).T, np.array([[0.0]])], axis=0),
                np.concatenate([np.atleast_2d(plate_orto_tf_y).T, np.array([[0.0]])], axis=0),
                np.concatenate([np.atleast_2d(plate_normal).T, np.array([[0.0]])], axis=0),
                np.concatenate([np.atleast_2d(center).T, np.array([[1.0]])], axis=0)
            ], axis=1
        ), check=False
    )
    return plate_virtual_tf

class AirhockeyConstraint(ob.Constraint):
    def __init__(self, sim: PyBulletWorld, robot_name: str, target_plane_state: EEState):
        self.__sim = sim
        self.__robot_name = robot_name
        self.__robot = self.__sim.get_robot(self.__robot_name)
        self.__codim = 3

        self.__ambient_dim = self.__robot.num_joints
        self.__q = np.zeros(self.__ambient_dim)

        # Choosing z-axis, rx and ry displacements in plane basis
        self.__selection_matrix_in_plate_basis = np.array([[0, 0, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]])
        self.__target_plane_state = target_plane_state

        super(AirhockeyConstraint, self).__init__(self.__ambient_dim, self.__codim)

    def function(self, state, out):
        ompl2numpy(state, self.__q)
        q1 = self.__q
        self.__robot.reset_joint_state(JointState.from_position(q1))
        striker_tf = self.__robot.ee_state(self.__target_plane_state.ee_link, self.__target_plane_state.ref_frame).tf
        err_tf = striker_tf@self.__target_plane_state.tf.inv()
        displacement = self.__selection_matrix_in_plate_basis@err_tf.twist().A
        
        numpy2ompl(displacement, out)

    def jacobian(self, state, out):
        ompl2numpy(state, self.__q)
        q1 = self.__q
        self.__robot.reset_joint_state(JointState.from_position(q1))
        J1 = self.__robot.jacobian(q1, self.__target_plane_state.ee_link, self.__target_plane_state.ref_frame)
        Jc = self.__selection_matrix_in_plate_basis @ J1
        for i in range(0, Jc.shape[0]):
            for j in range(0, Jc.shape[1]):
                out[i, j] = Jc[i, j]
    
    def distance(self, q):
        out = np.zeros(self.__codim)
        self.function(q, out)
        return np.linalg.norm(out)

    def project(self, projection):
        i = 0
        q = np.zeros(self.getAmbientDimension())
        ompl2numpy(projection, q)
        x = np.zeros(self.getCoDimension())
        J = np.zeros((self.getCoDimension(), self.getAmbientDimension()))
        self.function(q, x)
        # print(self.getTolerance())
        while x.dot(x)>= self.getTolerance()**2:
            if i > self.getMaxIterations():
                return False
            self.jacobian(q, J)
            q = q - np.linalg.pinv(J)@x
            self.function(q, x)
            i+=1
        numpy2ompl(q, projection)
        return True
    
    def project(self, projection):
        i = 0
        q = np.zeros(self.getAmbientDimension())
        ompl2numpy(projection, q)
        x = np.zeros(self.getCoDimension())
        J = np.zeros((self.getCoDimension(), self.getAmbientDimension()))
        self.function(q, x)
        # print(self.getTolerance())
        while x.dot(x)>= self.getTolerance()**2:
            if i > self.getMaxIterations():
                return False
            self.jacobian(q, J)
            q = q - np.linalg.pinv(J)@x
            self.function(q, x)
            i+=1
        numpy2ompl(q, projection)
        return True

    @property
    def target_plane_tf(self):
        return self.__target_plane_state.tf


class AirhockeyScene(BenchmarkConstrainedScene):
    def __init__(self, urdf_filename_robot1: str, table_link_name: str):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.SIMPLE_GUI, time_step = 0.00000001)
        self.__sim.add_object('airhockey_table', 'tasks/models/urdf/airhockey_table.urdf', fixed =True, save=True)
        self.__robot = self.__sim.add_robot(urdf_filename_robot1, SE3(-1.3,0, 0.00), 'robot1')
        self.__striker_link = 'iiwa_1/striker_mallet_tip'
        # self.__sim.sim_step()

        self.__state_space = ob.RealVectorStateSpace(self.__robot.num_joints)     
        bounds = ob.RealVectorBounds(self.__robot.num_joints)
        lb = self.__robot.joint_limits.limit_positions[0]
        ub = self.__robot.joint_limits.limit_positions[1]
        for i in range(self.__robot.num_joints):
            bounds.setLow(i, lb[i])
            bounds.setHigh(i, ub[i])
        self.__state_space.setBounds(bounds)

        self.__target_state = EEState.from_tf(SE3.Rx(np.pi), self.__striker_link, table_link_name)

        self.__constraint = AirhockeyConstraint(
            self.__sim,
            'robot1',
            self.__target_state
        )
    
    def get_constrained_configuration_from_workspace(
        self,
        tf_robot: EEState,
        initial_q: np.ndarray = None
    ):  
        if not initial_q is None:
            q1 = initial_q
            self.__robot.reset_joint_state(JointState.from_position(q1))
        self.__robot.reset_ee_state(EEState.from_tf(tf_robot, self.__target_state.ee_link, self.__target_state.ref_frame))
        # time.sleep(10)
        q1 = self.__robot.joint_state.joint_positions
        new_q = np.copy(q1)
        self.__constraint.project(new_q)
        return new_q
    
    def get_workspace_from_configuration( self, initial_q: np.ndarray) -> EEState:
        q1 = initial_q
        self.__robot.reset_joint_state(JointState.from_position(q1))
        return self.__robot.ee_state(self.__target_state.ee_link, self.__target_state.ref_frame)
    
    @property
    def state_space(self):
        return self.__state_space
    
    def is_q_valid(self, q) -> bool:
        q1 = q
        self.__robot.reset_joint_state(JointState.from_position(q1))
        if len(self.__sim.is_collide_with('robot1'))>0:
            return False
        striker_pose = self.__robot.ee_state(self.__target_state.ee_link).tf.t
        # print(striker_pose)
        if striker_pose[0]<-1 or striker_pose[1]>0.55 or striker_pose[1]<-0.55:
            return False
        return True
        
    @property
    def constraint(self)->ob.Constraint:
        return self.__constraint