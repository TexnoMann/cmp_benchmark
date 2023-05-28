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

def planning_once_by_planner(problem: ConstrainedProblem, planner_name: str, start_near_constraint: np.ndarray, goal_near_constraint: np.ndarray):
    problem.set_planner(planner_name)
    problem.set_start_and_goal(start_near_constraint, goal_near_constraint)
    # Solve the problem
    stat = problem.solve_once("dual_arm")
    return stat


def evaluate_planning(options):
    spaces = options.space.split(",")
    planners = options.planner.split(",")

    over_spaces_benchmark_results = {}
    for s in spaces:
        scene = DualArmScene('tasks/models/urdf/ur5e_pybullet.urdf', 'tasks/models/urdf/ur5e_pybullet.urdf')
        cp = ConstrainedProblem(s, scene, options)

        init_approx_q = np.array([-np.pi/2, -np.pi/2, np.pi/2, -np.pi, np.pi/2, 0.0, np.pi/2, -np.pi/2, -np.pi/2, 0.0, -np.pi/2, np.pi/2])
        start_robot1_ee_tf = SE3(-0.6, -0.15, 0.8) @ SE3.Rx(np.pi/2)
        end_robot1_ee_tf = SE3(-0.1, 0.1, 1.5) @ SE3.Rx(np.pi/2)
        q_start = scene.get_constrained_configuration_from_workspace(start_robot1_ee_tf, init_approx_q)
        q_end = scene.get_constrained_configuration_from_workspace(end_robot1_ee_tf)

        planning_results =  pd.DataFrame(columns = ["ok", "exec_time", "deviation"])
        for i in range(0, 1):
            result = planning_once_by_planner(cp, planners[0], q_start, q_end)
            planning_results.loc[i] = [result['ok'], result['exec_time'], result['deviation']]
            print("EVALUATE ITERATION {}".format(i))
        over_spaces_benchmark_results[s] = planning_results
        with open('dual_arm_{}_{}.pickle'.format(options.output, s), 'wb') as handle:
            pickle.dump(over_spaces_benchmark_results, handle, protocol=pickle.HIGHEST_PROTOCOL)




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