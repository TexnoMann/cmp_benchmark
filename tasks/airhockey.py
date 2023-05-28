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

class AirhockeyConstraint(ob.Constraint):
    def __init__(self, sim: PyBulletWorld, robot_name: str, stick_link_name: str, plate_pose: np.ndarray, plate_normal: np.ndarray):
        self.__sim = sim
        self.__robot_name = robot_name
        self.__robot = self.__sim.get_robot(self.__robot_name)
        self.__codim = 2

        self.__ambient_dim = self.__robot.num_joints
        self.__q = np.zeros(self.__ambient_dim)

        self.__plate_pose = plate_pose
        self.__selection_matrix = np.array([[1, 0, 0], [0, 1, 0]])

        self.__plate_normal = plate_normal/np.linalg.norm(plate_normal)
        self.__plate_orto_tf_x = np.cross(plate_normal, np.array([1,0,0]))
        self.__plate_orto_tf_y = np.cross(plate_normal, self.__plate_orto_tf_x)
        self.__plate_virtual_orient = np.concatenate(
            [   np.atleast_2d(self.__plate_orto_tf_x).T,
                np.atleast_2d(self.__plate_orto_tf_y).T,
                np.atleast_2d(self.__plate_normal).T
            ], axis=1
        )

        super(AirhockeyConstraint, self).__init__(self.__ambient_dim, self.__codim)

    def function(self, state, out):
        ompl2numpy(state, self.__q)
        q1 = self.__q
        self.__robot.reset_joint_state(JointState.from_position(q1))
        stick_tf = self.__robot.ee_state("tool_stick").tf
        pose = stick_tf.t
        displacement_pose = self.__plate_normal.dot(self.__plate_pose - pose)
        displacement_rotation = self.__selection_matrix @ self.__plate_virtual_orient.T@(SE3(SO3(stick_tf.R,check=False)) @ SE3(SO3(self.__plate_virtual_orient, check=False)).inv()).twist().A
        displacement = np.concatenate([displacement_pose, displacement_rotation], axis=0)
        numpy2ompl(displacement, out)

    def jacobian(self, state, out):
        ompl2numpy(state, self.__q)
        q1 = self.__q
        self.__robot.reset_joint_state(JointState.from_position(q1))
        J1 = self.__robot.jacobian(q1, "tool_stick")
        Rblock = np.kron(np.eye(2,dtype=float), self.__plate_virtual_orient.T)
        Jc = Rblock @ J1
        for i in range(0, Jc.shape[0]):
            for j in range(0, Jc.shape[1]):
                out[i, j] = Jc[i, j]
    
    def distance(self, q):
        out = np.zeros(self.__codim)
        self.function(q, out)
        return np.linalg.norm(out)

    def project(self, state, projection):
        i = 0
        q = np.zeros(self.getAmbientDimension())
        ompl2numpy(state, q)
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
        print("Project")
        return True


class DualArmScene(BenchmarkConstrainedScene):
    def __init__(self, urdf_filename_robot1: str, urdf_filename_robot2: str):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.SIMPLE_GUI, time_step = 0.01)
        self.__sim.add_object('table', 'tasks/models/urdf/table.urdf', fixed =True, save=True)
        self.__robot = self.__sim.add_robot(urdf_filename_robot1, SE3(0,-0.3,0.655), 'robot1')

        # self.__robot.connect_tool('peg' ,'tests/urdf/hole_round.urdf', root_link='ee_tool', tf=SE3(0.0, 0.0, 0.1))
        self.__sim.sim_step()

        self.__state_space = ob.RealVectorStateSpace(self.__robot.num_joints + self.__robot2.num_joints)     
        bounds = ob.RealVectorBounds(self.__robot.num_joints + self.__robot2.num_joints)
        lb = np.concatenate([self.__robot.joint_limits.limit_positions[0], self.__robot2.joint_limits.limit_positions[0]])
        ub = np.concatenate([self.__robot.joint_limits.limit_positions[1], self.__robot2.joint_limits.limit_positions[1]])
        for i in range(self.__robot.num_joints + self.__robot2.num_joints):
            bounds.setLow(i, lb[i])
            bounds.setHigh(i, ub[i])
        self.__state_space.setBounds(bounds)

        self.__constraint = DualArmConstraint(
            self.__sim,
            'robot1',
            'robot2',
            EEState.from_tf(SE3(0.0, 0.0, 0.1)@SE3.Ry(np.pi),"tool0", "tool0")
        )
    
    def get_constrained_configuration_from_workspace(
        self,
        tf_robot2: EEState,
        initial_q: np.ndarray = None
    ):  
        if not initial_q is None:
            q1 = initial_q[:self.__robot.num_joints]
            q2 = initial_q[self.__robot.num_joints:]
            self.__robot.reset_joint_state(JointState.from_position(q1))
            self.__robot2.reset_joint_state(JointState.from_position(q2))
        # time.sleep(10)
        self.__robot2.reset_ee_state(EEState.from_tf(tf_robot2, "tool0", "world"))
        self.__robot.reset_ee_state(EEState.from_tf(tf_robot2@self.__constraint.target_tf.tf, "tool0", "world"))
        q1 = self.__robot.joint_state.joint_positions
        q2 = self.__robot2.joint_state.joint_positions
        q = np.concatenate([q1, q2])
        new_q = np.copy(q)
        self.__constraint.project(q, new_q)
        return new_q
    
    def get_workspace_from_configuration( self, initial_q: np.ndarray) -> EEState:
        q1 = initial_q[:self.__robot.num_joints]
        q2 = initial_q[self.__robot.num_joints:]
        self.__robot.reset_joint_state(JointState.from_position(q1))
        return self.__robot.ee_state("tool0")
    
    @property
    def state_space(self):
        return self.__state_space
    
    def is_q_valid(self, q) -> bool:
        q1 = q[:self.__robot.num_joints]
        q2 = q[self.__robot.num_joints:]
        self.__robot.reset_joint_state(JointState.from_position(q1))
        self.__robot2.reset_joint_state(JointState.from_position(q2))
        if len(self.__sim.is_collide_with('robot1'))>0:
            return False
        if len(self.__sim.is_collide_with('robot2'))>0:
            return False
        return True
        
    @property
    def constraint(self)->ob.Constraint:
        return self.__constraint