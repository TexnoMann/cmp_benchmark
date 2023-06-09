from __future__ import print_function
import argparse
import pandas as pd
import math
import time
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.benchmark import *
from spatialmath import SE3
from itmobotics_sim.utils.robot import EEState
import pickle

from tasks.dual_arm import DualArmScene

N_RAND_INIT_Q = 100
N_PLAN_ITER = 20
ALGORITHM_NAMES = {'PJ': 'CBiRRT', 'TB': 'TBRRT', 'AT': 'AtlasRRT'}

def planning_once_by_planner(problem: ConstrainedProblem, planner_name: str, start_near_constraint: np.ndarray, goal_near_constraint: np.ndarray):
    problem.set_planner(planner_name)
    problem.set_start_and_goal(start_near_constraint, goal_near_constraint)
    # Solve the problem
    stat = problem.solve_once("dual_arm")
    return stat


def evaluate_planning(options):
    opt_vars = vars(options)
    print(opt_vars)
    spaces = options.space
    planners = options.planner

    benchmark_results = pd.DataFrame(columns = ["algorithm", "planner", "exec_time", "ok", "deviation"])

    base_pose = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])

    random_init_q = np.random.uniform(-0.1, 0.1, (N_RAND_INIT_Q, 12)) + np.concatenate([base_pose, base_pose], axis=0)

    start_robot1_ee_tf = SE3(-0.6, -0.1, 0.7) @ SE3.Rx(np.pi/2)
    # end_robot1_ee_tf = SE3(0.4, -0.0, 0.8) @ SE3.Rx(np.pi/2)
    end_robot1_ee_tf = SE3(-0.1, 0.1, 0.9) @ SE3.Rx(np.pi/2)
    
    scene = DualArmScene('tasks/models/urdf/ur5e_pybullet.urdf', 'tasks/models/urdf/ur5e_pybullet2.urdf')
    # scene.sim.registrate_blender_objects()
    for j in range(0, N_RAND_INIT_Q):
        q_start = scene.get_constrained_configuration_from_workspace(start_robot1_ee_tf, random_init_q[j,:])
        # time.sleep(20)
        q_end = scene.get_constrained_configuration_from_workspace(end_robot1_ee_tf, q_start)
        # time.sleep(20)
        # scene.sim.sim_step()
        if not (scene.is_q_valid(q_start) and scene.is_q_valid(q_end)):
            continue
        scene.sim.sim_step()
        for s in spaces:
            print("START PLAN WITH {} SPACE".format(s))
            cp = ConstrainedProblem(s, scene, options)
            for i in range(0, N_PLAN_ITER):
                print("EVALUATE ITERATION {}".format(i))
                # time.sleep(10)
                for p in planners:
                    scene.sim.sim_step()
                    result = planning_once_by_planner(cp, p, q_start, q_end)
                    # scene.sim.sim_step()
                    # scene.sim.save_blender_date('dual_arm_scene.pkl')
                    group_out_result = [ALGORITHM_NAMES[s], p, result['exec_time'], result['ok'], result['deviation']]
                    print("Write to dataframe planning info: {}".format(group_out_result))
                    benchmark_results.loc[len(benchmark_results.index)] = group_out_result
            del cp
        break
    del scene

    file_name = 'dual_arm_{}_e{}_d{}.pickle'.format(options.output, options.epsilon, options.delta, planners[0])
    with open(file_name, 'wb') as handle:
        pickle.dump({"result": benchmark_results, "options": options}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("WRITING RESULT INTO: {}".format(file_name))
    time.sleep(4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true",
                        help="Do benchmarking on provided planner list.")
    addSpaceOption(parser)
    addPlannerOption(parser)
    addConstrainedOptions(parser)
    addAtlasOptions(parser)
    addInputOutputOption(parser)

    evaluate_planning(parser.parse_args())