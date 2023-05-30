from __future__ import print_function
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import tools as ot
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys
    sys.path.insert(
        0, join(dirname(dirname(dirname(abspath(__file__)))), 'py-bindings'))
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import tools as ot
import datetime
import pandas as pd
import os, sys
import time

from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prepare_results import *
from benchmark.benchmark_scene import BenchmarkConstrainedScene, list2vec, ompl2numpy,numpy2ompl, NewtonRaphsonProjectionEvaluator

def addSpaceOption(parser):
    parser.add_argument("-s", "--space", default=["PJ"],
        choices=["PJ", "AT", "TB"], nargs="+",
        help="""Choose which constraint_function handling methodology to use. 
        One of:
            PJ - Projection (Default)
            AT - Atlas
            TB - Tangent Bundle.
        """
    )

def addInputOutputOption(parser):
    parser.add_argument("-o", "--output", type=str, default="result",
        help="""Choose output filename"""
    )

def addPlannerOption(parser):
    parser.add_argument("-p", "--planner", default=["RRTConnect"],
        choices=["RRT", "RRTConnect", "RRTstar", "EST", "BiEST", "ProjEST", "BITstar", "PRM", "KPIECE1", "BKPIECE1"],
        nargs="+",
        help="List of which motion planner to use.\n "
        "Choose from, e.g.:\n"
        "RRT, RRTConnect (Default), RRTstar, "
        "EST, BiEST, ProjEST, "
        "BITstar, "
        "PRM, SPARS, "
        "KPIECE1, BKPIECE1."
    )


def addConstrainedOptions(parser):
    group = parser.add_argument_group("Constrained planning options")
    group.add_argument("-d", "--delta", type=float, default=ob.CONSTRAINED_STATE_SPACE_DELTA,
                    help="Step-size for discrete geodesic on manifold.")
    group.add_argument("--lambda", type=float, dest="lambda_", metavar="LAMBDA",
                    default=ob.CONSTRAINED_STATE_SPACE_LAMBDA,
                    help="Maximum `wandering` allowed during atlas traversal. Must be greater "
                    "than 1.")
    group.add_argument("--tolerance", type=float, default=0.05,
                    help="constraint_function satisfaction tolerance.")
    group.add_argument("--time", type=float, default=100.,
                    help="Planning time allowed.")
    group.add_argument("--projector_method", type=str, default="NewtonRaphsonProjection",
                    help="Projection method.")
    group.add_argument("--tries", type=int, default=ob.CONSTRAINT_PROJECTION_MAX_ITERATIONS,
                    help="Maximum number sample tries per sample.")
    group.add_argument("-r", "--range", type=float, default=0.,
                    help="Planner `range` value for planners that support this parameter. "
                    "Automatically determined otherwise (when 0).")

def clearSpaceAndPlanner(planner):
    planner.getSpaceInformation().getStateSpace().clear()
    planner.clear()


def addAtlasOptions(parser):
    group = parser.add_argument_group("Atlas options")
    group.add_argument("--epsilon", type=float, default=ob.ATLAS_STATE_SPACE_EPSILON,
                    help="Maximum distance from an atlas chart to the manifold. Must be "
                    "positive.")
    group.add_argument("--rho", type=float, default=ob.CONSTRAINED_STATE_SPACE_DELTA *
                    ob.ATLAS_STATE_SPACE_RHO_MULTIPLIER,
                    help="Maximum radius for an atlas chart. Must be positive.")
    group.add_argument("--exploration", type=float, default=ob.ATLAS_STATE_SPACE_EXPLORATION,
                    help="Value in [0, 1] which tunes balance of refinement and exploration in "
                    "atlas sampling.")
    group.add_argument("--alpha", type=float, default=ob.ATLAS_STATE_SPACE_ALPHA,
                    help="Maximum angle between an atlas chart and the manifold. Must be in "
                    "[0, PI/2].")
    group.add_argument("--bias", action="store_true",
                    help="Sets whether the atlas should use frontier-biased chart sampling "
                    "rather than uniform.")
    group.add_argument("--no-separate", action="store_true",
                    help="Sets that the atlas should not compute chart separating halfspaces.")
    group.add_argument("--charts", type=int, default=ob.ATLAS_STATE_SPACE_MAX_CHARTS_PER_EXTENSION,
                    help="Maximum number of atlas charts that can be generated during one "
                    "manifold traversal.")


class ConstrainedProblem(object):

    def __init__(self, planning_space_type: str, scene: BenchmarkConstrainedScene, options):
        self.__planning_space_type = planning_space_type
        self.__space = scene.state_space
        self.__constraint = scene.constraint
        self.__constraint.setTolerance(options.tolerance)
        self.__constraint.setMaxIterations(options.tries)
        self.__validation_function = ob.StateValidityCheckerFn(scene.is_state_valid)
        self.options = options
        self.__planer = None

        if planning_space_type == "PJ":
            ou.OMPL_INFORM("Using Projection-Based State Space!")
            self.__css = ob.ProjectedStateSpace(self.__space, self.__constraint)
            self.__csi = ob.ConstrainedSpaceInformation(self.__css)
        elif planning_space_type == "AT":
            ou.OMPL_INFORM("Using Atlas-Based State Space!")
            self.__css = ob.AtlasStateSpace(self.__space, self.__constraint)
            self.__csi = ob.ConstrainedSpaceInformation(self.__css)
        elif planning_space_type == "TB":
            ou.OMPL_INFORM("Using Tangent Bundle-Based State Space!")
            self.__css = ob.TangentBundleStateSpace(self.__space, self.__constraint)
            self.__csi = ob.TangentBundleSpaceInformation(self.__css)

        self.__css.setup()
        self.__css.setDelta(options.delta)
        self.__css.setLambda(options.lambda_)
        if not planning_space_type == "PJ":
            self.__css.setExploration(options.exploration)
            self.__css.setEpsilon(options.epsilon)
            self.__css.setRho(options.rho)
            self.__css.setAlpha(options.alpha)
            self.__css.setMaxChartsPerExtension(options.charts)
            if options.bias:
                self.__css.setBiasFunction(lambda c, atlas=self.__css: atlas.getChartCount() - c.getNeighborCount() + 1.)
            if planning_space_type == "AT":
                self.__css.setSeparated(not options.no_separate)
            self.__css.setup()
        self.__ss = og.SimpleSetup(self.__csi)

    def set_start_and_goal(self, q_start: np.ndarray, q_end: np.ndarray):
        proj_q_start = np.copy(q_start)
        proj_q_end = np.copy(q_end)
        self.__constraint.project(proj_q_start)
        self.__constraint.project(proj_q_end)
        self.__ss.clear()
        start = ob.State(self.__css)
        goal = ob.State(self.__css)
        for i in range(self.__space.getDimension()):
            start[i] = proj_q_start[i]
            goal[i] = proj_q_end[i]
            
        # Create start and goal states
        if self.__planning_space_type == "AT" or self.__planning_space_type == "TB":
            self.__css.anchorChart(start())
            self.__css.anchorChart(goal())

        # Setup problem
        self.__ss.setStartAndGoalStates(start, goal)
        self.__ss.setStateValidityChecker(self.__validation_function)

    def get_planner_by_name(self, planner_name: str):
        csi = self.__csi
        planner = eval('og.%s(csi)' % planner_name)
        try:
            if self.options.range == 0:
                if not self.__planning_space_type == "PJ":
                    planner.setRange(self.__css.getRho_s())
            else:
                planner.setRange(self.options.range)
        except:
            pass
        return planner

    def set_planner(self, planner_name: str):
        self.__planer = self.get_planner_by_name(planner_name)
        self.__ss.setPlanner(self.__planer)

    def solve_once(self, name="ompl"):
        self.__ss.setup()
        start_time = time.time()
        ok = True
        stat = self.__ss.solve(self.options.time)
        if stat:
            # Get solution and validate
            try:
                # Prevent PRM error
                time.sleep(1)
                path = self.__ss.getSolutionPath()
            except:
                stat = False
        end_time = time.time()
        if stat:
            if not path.check():
                ou.OMPL_WARN("Path fails check!")
                ok = False

            if stat == ob.PlannerStatus.APPROXIMATE_SOLUTION:
                ou.OMPL_WARN("Solution is approximate.")

            # Simplify solution and validate simplified solution path.
            ou.OMPL_INFORM("Simplifying solution...")
            self.__ss.simplifySolution(60.)

            simplePath = self.__ss.getSolutionPath()
            ou.OMPL_INFORM("Simplified Path Length: %.3f -> %.3f" %
                        (path.length(), simplePath.length()))

            if not simplePath.check():
                ou.OMPL_WARN("Simplified path fails check!")

            ou.OMPL_INFORM("Interpolating simplified path...")
            simplePath.interpolate(1000)

            if not simplePath.check():
                ou.OMPL_WARN("Interpolated simplified path fails check!")
                simplePath = path

            ou.OMPL_INFORM("Inerpolated Path Contain: %d points"%(simplePath.getStateCount()))
                
        else:
            ou.OMPL_WARN("No solution found.")
            ok=False
        
        if ok and stat:
            states = [[x[i] for i in range(self.__css.getAmbientDimension())] for x in simplePath.getStates()]
            states_array = np.asarray(states)
            deviation = calc_constraint_deviation(states_array, self.__constraint)
            return {"path": states_array, "deviation": deviation, "exec_time": (end_time-start_time), "ok": int(ok), "init_ok": stat}
        
        return {"exec_time": (end_time-start_time), "ok": int(ok), "deviation": None, "init_ok": stat}

    def atlas_stats(self):
        # For atlas types, output information about size of atlas and amount of
        # space explored
        if self.__planning_space_type == "AT" or self.__planning_space_type == "TB":
            ou.OMPL_INFORM("Atlas has %d charts" % self.__css.getChartCount())
            if self.__planning_space_type == "AT":
                ou.OMPL_INFORM("Atlas is approximately %.3f%% open" %
                            self.__css.estimateFrontierPercent())

    def dump_graph(self, name):
        ou.OMPL_INFORM("Dumping planner graph to `%s_graph.graphml`." % name)
        data = ob.PlannerData(self.__csi)
        self.__planer.getPlannerData(data)

        with open("logs/%s_graph.graphml" % name, "w") as graphfile:
            print(data.printGraphML(), file=graphfile)

        if self.__planning_space_type == "AT" or self.__planning_space_type == "TB":
            ou.OMPL_INFORM("Dumping atlas to `%s_atlas.ply`." % name)
            with open("logs/%s_atlas.ply" % name, "w") as atlasfile:
                print(self.__css.printPLY(), file=atlasfile)