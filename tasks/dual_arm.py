import os, sys

import numpy as np
from ompl import base as ob
from ompl import geometric as og

import os
import sys
import time

import numpy as np
from spatialmath import SE3
from spatialmath import base as sb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.benchmark_scene import BenchmarkConstrainedScene
from benchmark.benchmark import ompl2numpy, numpy2ompl

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot
from itmobotics_sim.utils.controllers import EEPositionToEEVelocityController, EEVelocityToJointVelocityController, JointTorquesController

class DualArmConstraint(ob.Constraint):
    def __init__(self, sim: PyBulletWorld, robot_name1: str, robot_name2: str, target_tf: EEState):
        self.__sim = sim
        self.__robot_name1 = robot_name1
        self.__robot_name2 = robot_name2
        self.__target_tf = target_tf
        self.__robot = self.__sim.get_robot(robot_name1)
        self.__robot2 = self.__sim.get_robot(robot_name2)
        self.__codim = 6

        self.__ambient_dim = self.__robot.num_joints + self.__robot2.num_joints
        self.__q = np.zeros(self.__ambient_dim)

        super(DualArmConstraint, self).__init__(self.__ambient_dim, self.__codim)

    def function(self, state, out):
        ompl2numpy(state, self.__q)
        q1 = self.__q[:self.__robot.num_joints]
        q2 = self.__q[self.__robot.num_joints:]
        self.__robot.reset_joint_state(JointState.from_position(q1))
        self.__robot2.reset_joint_state(JointState.from_position(q2))

        tf_btw_ee_in_ee2_frame = self.__sim.link_state(
            self.__robot_name1,
            self.__target_tf.ee_link,
            self.__robot_name2,
            self.__target_tf.ref_frame
        )
        displacement = (tf_btw_ee_in_ee2_frame.tf@self.__target_tf.tf.inv()).twist().A
        numpy2ompl(displacement, out)

    def jacobian(self, state, out):
        ompl2numpy(state, self.__q)
        q1 = self.__q[:self.__robot.num_joints]
        q2 = self.__q[self.__robot.num_joints:]
        self.__robot.reset_joint_state(JointState.from_position(q1))
        self.__robot2.reset_joint_state(JointState.from_position(q2))
        robot2_tf = self.__robot2.ee_state(self.__target_tf.ref_frame, "world").tf
        self.__sim.sim_step()
        J1 = self.__robot.jacobian(q1, self.__target_tf.ee_link, "world")
        J2 = self.__robot2.jacobian(q2, self.__target_tf.ref_frame, "world")
        Rblock = np.kron(np.eye(2,dtype=float), robot2_tf.R.T)
        Jc = Rblock@np.concatenate((J1, -J2), axis=1)
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
        while x.dot(x)>= self.getTolerance()**2:
            if i > self.getMaxIterations():
                return False
            self.jacobian(q, J)
            q = q - np.linalg.pinv(J)@x
            q1 = q[:self.__robot.num_joints]
            q2 = q[self.__robot.num_joints:]
            self.__robot.reset_joint_state(JointState.from_position(q1))
            self.__robot2.reset_joint_state(JointState.from_position(q2))
            if len(self.__sim.is_collide_with('robot1'))>0:
                return False
            if len(self.__sim.is_collide_with('robot2'))>0:
                return False
            self.function(q, x)
            i+=1
        numpy2ompl(q, projection)
        return True

    @property
    def target_tf(self):
        return self.__target_tf


class DualArmScene(BenchmarkConstrainedScene):
    def __init__(self, urdf_filename_robot1: str, urdf_filename_robot2: str):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.SIMPLE_GUI, time_step = 0.0000001)
        self.__sim.add_object('table', 'tasks/models/urdf/table.urdf', fixed =True, save=True)
        self.__sim.add_object('block', 'tasks/models/urdf/block.urdf', fixed =True, save=True, base_transform=SE3(-0.65, 0.0, 0.95))
        self.__robot = self.__sim.add_robot(urdf_filename_robot1, SE3(0,-0.3,0.655), 'robot1')
        self.__robot2 = self.__sim.add_robot(urdf_filename_robot2, SE3(0.0,0.3,0.655), 'robot2')

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
        self.__constraint.project(new_q)
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
        object_pose = self.__robot.ee_state("tool0").tf.t
        if object_pose[2]<0.65:
            return False
        return True
        
    @property
    def constraint(self)->ob.Constraint:
        return self.__constraint

if __name__ == "__main__":
    scene = DualArmScene('models/urdf/ur5e_pybullet.urdf', 'models/urdf/ur5e_pybullet.urdf')
