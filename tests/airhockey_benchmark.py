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

from tasks.airhockey import AirhockeyScene

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

    random_init_q = np.random.uniform(-np.pi+0.0001, np.pi-0.0001, (N_RAND_INIT_Q, 12))
    random_end_q = np.random.uniform(-np.pi+0.0001, np.pi-0.0001, (N_RAND_INIT_Q, 12))
    start_robot1_ee_tf = SE3(-0.6, -0.15, 0.8) @ SE3.Rx(np.pi/2)@SE3.Ry(np.pi/8)
    end_robot1_ee_tf = SE3(-0.1, 0.1, 1.4) @ SE3.Rx(np.pi/2)
    
    for s in spaces:
        print("START PLAN WITH {} SPACE".format(s))
        q_ok = False
        scene = AirhockeyScene('tasks/models/urdf/iiwa_airhockey.urdf', "iiwa_1/link_0")
        cp = ConstrainedProblem(s, scene, options)
        for j in range(0, N_RAND_INIT_Q):
            if q_ok:
                break
            for k in range(0, N_RAND_INIT_Q):
                q_start = scene.get_constrained_configuration_from_workspace(start_robot1_ee_tf, random_init_q[j,:])
                q_end = scene.get_constrained_configuration_from_workspace(end_robot1_ee_tf, random_end_q[k,:])
                if not (scene.is_q_valid(q_start) and scene.is_q_valid(q_end)):
                    continue
                for i in range(0, N_PLAN_ITER):
                    print("EVALUATE ITERATION {}".format(i))
                    # time.sleep(10)
                    for p in planners:
                        result = planning_once_by_planner(cp, p, q_start, q_end)
                        group_out_result = [ALGORITHM_NAMES[s], p, result['exec_time'], result['ok'], result['deviation']]
                        print("Write to dataframe planning info: {}".format(group_out_result))
                        benchmark_results.loc[len(benchmark_results.index)] = group_out_result
                q_ok = True
                break
        del scene, cp

    with open('airhockey_{}_e{}_d{}.pickle'.format(options.output, options.epsilon, options.delta, planners[0]), 'wb') as handle:
        pickle.dump(benchmark_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    addSpaceOption(parser)
    addPlannerOption(parser)
    addConstrainedOptions(parser)
    addAtlasOptions(parser)
    addInputOutputOption(parser)

    evaluate_planning(parser.parse_args())