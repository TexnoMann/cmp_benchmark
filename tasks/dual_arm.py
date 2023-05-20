import os, sys

import numpy as np
from ompl import base as ob
from ompl import geometric as og

import os
import sys
import copy
import unittest

import numpy as np
from spatialmath import SE3
from spatialmath import base as sb

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot
from itmobotics_sim.utils.controllers import EEPositionToEEVelocityController, EEVelocityToJointVelocityController, JointTorquesController


target_ee_state = EEState.from_tf( SE3(0.3, -0.5, 1.2) @ SE3.Rx(np.pi) , 'ee_tool')
target_ee_state.twist = np.array([0,0,0.01,0,0,0])

target_ee_state2 = EEState.from_tf(SE3(0.3, 0.1, 1.2) @ SE3.Rx(np.pi), 'ee_tool')
target_ee_state2.twist = np.array([0,0,-0.01,0,0,0])


target_joint_state = JointState.from_position(np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]))
target_joint_state = JointState.from_torque(np.zeros(6))

target_pose_motion = Motion.from_states(target_joint_state,target_ee_state)
target_speed_motion = copy.deepcopy(target_pose_motion)

class DualArmScene(BenchmarkConstrainedScene):
    def __init__(self, urdf_filename_robot1: str, urdf_filename_robot2: str):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.DIRECT, time_step = 0.01)
        self.__sim.add_object('table', 'tests/urdf/table.urdf', fixed =True, save=True)
        self.__robot = self.__sim.add_robot(urdf_filename_robot1, SE3(0,-0.3,0.625), 'robot1')
        self.__robot2 = self.__sim.add_robot(urdf_filename_robot2, SE3(0.0,0.3,0.625), 'robot2')

        self.__robot.connect_tool('peg' ,'tests/urdf/hole_round.urdf', root_link='ee_tool', tf=SE3(0.0, 0.0, 0.1))
        self.__sim.sim_step()

        self.__state_space = ob.RealVectorStateSpace(self.__robot.num_joints + self.__robot2.num_joints)     
        bounds = ob.RealVectorBounds(self.__robot.num_joints + self.__robot2.num_joints)
        lb = np.concatenate([self.__robot.joint_limits.limit_positions[0], self.__robot2.joint_limits.limit_positions[0]])
        ub = np.concatenate([self.__robot.joint_limits.limit_positions[1], self.__robot2.joint_limits.limit_positions[1]])
        for i in range(self.__robot.num_joints + self.__robot2.num_joints):
            bounds.setLow(i, lb[i])
            bounds.setHigh(i, ub[i])
        self.__state_space.setBounds(bounds)
    
    @property
    def state_space(self):
        return self.__state_space
