import os, sys

import numpy as np
from ompl import base as ob
from ompl import geometric as og
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.air_hockey_challenge.air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from tasks.air_hockey_challenge.air_hockey_challenge.framework import AgentBase

import mujoco
from tasks.air_hockey_challenge.air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian, link_to_xml_name
from constraints.airhockey_ompl_constraint import AirHockeyCircleConstraint


class CBiRRTPlanner:
    def __init__(self, env_info: dict, constraint: ob.Constraint):
        self.__ambient_dim = env_info['robot']['n_joints']

        self.time = 60.
        tolerance = 0.01#ob.CONSTRAINT_PROJECTION_TOLERANCE
        tries = ob.CONSTRAINT_PROJECTION_MAX_ITERATIONS
        lambda_ = ob.CONSTRAINED_STATE_SPACE_LAMBDA
        delta = ob.CONSTRAINED_STATE_SPACE_DELTA

        rvss = ob.RealVectorStateSpace(self.__ambient_dim)     
        bounds = ob.RealVectorBounds(self.__ambient_dim)
        lb = env_info['robot']['joint_pos_limit'][0,:]
        ub = env_info['robot']['joint_pos_limit'][1,:]
        for i in range(self.__ambient_dim):
            bounds.setLow(i, lb[i])
            bounds.setHigh(i, ub[i])
        rvss.setBounds(bounds)

        constraint.setTolerance(tolerance)
        constraint.setMaxIterations(tries)
        self.css = ob.ProjectedStateSpace(rvss, constraint)
        self.csi = ob.ConstrainedSpaceInformation(self.css)
        self.ss = og.SimpleSetup(self.csi)

        self.css.setDelta(delta)
        self.css.setLambda(lambda_)

        self.planner = og.RRTConnect(self.csi)
        self.ss.setPlanner(self.planner)

        self.css.setup()
        self.ss.setup()

    def in_obstacle(self, q) ->bool:
        pass

    def solve(self, q0, qk):
        self.ss.clear()
        start = ob.State(self.css)
        goal = ob.State(self.css)
        for i in range(self.__ambient_dim):
            start[i] = q0[i]
            goal[i] = qk[i]
        
        self.ss.setStartAndGoalStates(start, goal, 0.005)
        # self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(lambda x: self.in_obstacle(q))
        stat = self.ss.solve(self.time)
        planning_time = self.ss.getLastPlanComputationTime()
        success = False
        q = np.array([q0])
        t = np.array([0.])
        if stat:
            # Get solution and validate
            path = self.ss.getSolutionPath()
            print("interpolated path")
            print(path)
            states = [[x[i] for i in range(self.__ambient_dim)] for x in path.getStates()]
            q = np.array(states)
            success = True
            q_diff = q[1:] - q[:-1]
            diff = np.sum(np.abs(q_diff), axis=-1)
            include = np.concatenate([diff > 0, [True]])
            q = q[include]
            q_diff = q_diff[include[:-1]]
            ts = np.abs(q_diff) / 10
            t = np.max(ts, axis=-1)
            t = np.concatenate([[0.], t + 1e-4])
            t = np.cumsum(t)
        return q, [], [], t, planning_time

class DummyAgent(AgentBase):
    def __init__(self, env_info, value, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None

        self.primitive_variable = value  # Primitive python variable
        self.numpy_vector = np.array([1, 2, 3]) * value  # Numpy array
        self.list_variable = [1, 'list', [2, 3]]  # Numpy array

        # Dictionary
        self.dictionary = dict(some='random', keywords=2, fill='the dictionary')

        # Building a torch object
        data_array = np.ones(3) * value
        data_tensor = torch.from_numpy(data_array)
        self.torch_object = torch.nn.Parameter(data_tensor)

        # A non serializable object
        self.object_instance = object()

        # A variable that is not important e.g. a buffer
        self.not_important = np.zeros(10000)

        # Here we specify how to save each component
        self._add_save_attr(
            primitive_variable='primitive',
            numpy_vector='numpy',
            list_variable='primitive',
            dictionary='pickle',
            torch_object='torch',
            object_instance='none',
            # The '!' is to specify that we save the variable only if full_save is True
            not_important='numpy!',
        )

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        print(self.get_joint_pos(observation))
        if self.new_start:
            self.new_start = False
            self.hold_position = self.get_joint_pos(observation)

        velocity = np.zeros_like(self.hold_position)
        action = np.vstack([self.hold_position, velocity])
        return action

def main():
    import time
    np.random.seed(0)

    env = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity",
                                    interpolation_order=3, debug=False)
    agents = DummyAgent(env.env_info, 5)
    obs = env.reset()
    agents.episode_start()

    qs = agents.get_joint_pos(obs)
    current_ee_pose = forward_kinematics(
        env.env_info['robot']['robot_model'],
        env.env_info['robot']['robot_data'],
        qs
    )
    # print(current_ee_pose)
    qf = inverse_kinematics(
        env.env_info['robot']['robot_model'],
        env.env_info['robot']['robot_data'],
        current_ee_pose[0] + np.array([0.2, 0, 0]), initial_q=qs
    )
    print(current_ee_pose)

    planner = CBiRRTPlanner(
        env.env_info,
        AirHockeyCircleConstraint(
            env.env_info['robot']['n_joints'],
            env.env_info['robot']['robot_model'],
            env.env_info['robot']['robot_data'],
            current_ee_pose[0] + np.array([0.1, 0, 0]), 0.1)
    )
    result = planner.solve(qs, qf[1])[0]
    print(result)
    steps = 0
    while True:
        steps += 1
        t_start = time.time()
        if ((steps-1)/result.shape[0])%2 == 0:
            action = np.flip(result, axis=0)[(steps-1)%result.shape[0],:]
        else:
            action = result[(steps-1)%result.shape[0],:]
        
        joints_pose_vel = np.vstack([action,np.zeros(action.shape)])
        obs, reward, done, info = env.step(joints_pose_vel)

        env.render()

        if done or steps > env.info.horizon:
            from scipy.interpolate import CubicSpline
            import matplotlib.pyplot as plt
            ee_result = np.zeros(result.shape)
            print("EE Path:")
            for i in range(result.shape[0]):
                ee_pose = forward_kinematics(
                    env.env_info['robot']['robot_model'],
                    env.env_info['robot']['robot_data'],
                    result[i,:]
                )[0]
                print(ee_pose)
                ee_result[i,:] = ee_pose
            cs = CubicSpline(np.arange(0, ee_result.shape[0]), ee_result)
            xs = np.arange(0, ee_result.shape[0]-1, 0.1)

            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.plot(ee_result[:, 0], ee_result[:, 1], 'o', label='data')
            # ax.plot(np.cos(xs), np.sin(xs), label='true')
            ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
            ax.axes.set_aspect('equal') 
            ax.legend(loc='center')
            plt.show()
            # import matplotlib.pyplot as plt
            # import matplotlib
            # matplotlib.use("tkAgg")
            # trajectory_record = np.array(env.base_env.controller_record)
            # nq = env.base_env.env_info['robot']['n_joints']
            #
            # fig, axes = plt.subplots(3, nq)
            # for j in range(nq):
            #     axes[0, j].plot(trajectory_record[:, j])
            #     axes[0, j].plot(trajectory_record[:, j + nq])
            #     axes[1, j].plot(trajectory_record[:, j + 2 * nq])
            #     axes[1, j].plot(trajectory_record[:, j + 3 * nq])
            #     # axes[2, j].plot(trajectory_record[:, j + 4 * nq])
            #     axes[2, j].plot(trajectory_record[:, j + nq] - trajectory_record[:, j])
            # plt.show()

            steps = 0
            obs = env.reset()
            agents.episode_start()
            print("Reset")


if __name__ == '__main__':
    main()