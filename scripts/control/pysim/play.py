# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an algorithm."""

import argparse

# import numpy as np
import sys
import torch

from isaaclab.app import AppLauncher
import gymnasium as gym
import threading
import queue
from contextlib import suppress

parser = argparse.ArgumentParser(description="Train an RL agent with HARL.")

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_env_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--debug", action="store_true", help="whether to run in debug mode for visualization")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

agent_cfg_entry_point = f"pysim_integration_cfg_entry_point"

import torch
from py_sim.worlds.polygon_world import PolygonWorld
import py_sim.plotting.plotting as pt
import numpy as np
import matplotlib.pyplot as plt
from py_sim.tools.sim_types import TwoDimArray
from py_sim.isaaclab.sample_planning import run_rrt_planner
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import py_sim.dynamics.bicycle as bike
import py_sim.dynamics.differential_drive as diff
import py_sim.dynamics.unicycle as uni
import py_sim.sensors.occupancy_grid as og
import py_sim.worlds.polygon_world as poly_world
from matplotlib.axes import Axes
from py_sim.dynamics.unicycle import solution_trajectory as unicycle_solution_trajectory
from py_sim.path_planning import dwa
from py_sim.path_planning.path_generation import create_path
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.sensors.range_bearing import RangeBearingSensor
from py_sim.sim.generic_sim import SimParameters, start_sim
from py_sim.sim.sim_modes import (
    DwaFollower,
    DwaFollowerIsaacLab,
    NavVectorFollower,
    SimpleSim,
    VectorFollower,
)
from py_sim.tools.projections import LineCarrot
from py_sim.tools.sim_types import (
    ArcParams,
    DwaParams,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
)
from py_sim.vectorfield.vectorfields import (  # pylint: disable=unused-import
    AvoidObstacle,
    G2GAvoid,
    GoToGoalField,
    SummedField,
)

def grid_to_polygon_world_big_components(
    grid_tensor,
    cell_size: float = 1.0,
    origin=(0.0, 0.0),
    occupied_value=1,
):
    """
    Convert a 2D occupancy grid into a PolygonWorld with ONE rectangle per
    connected obstacle region (4-connected).

    - grid_tensor: torch.Tensor or np.ndarray of shape (H, W)
    - cell_size:  size of one cell in world units
    - origin:     (x0, y0) world coordinate corresponding to grid (row=0, col=0)
    - occupied_value: value in the grid treated as obstacle
    """
    # Convert to numpy if needed
    if isinstance(grid_tensor, torch.Tensor):
        grid = grid_tensor.cpu().numpy()
    else:
        grid = np.asarray(grid_tensor)

    H, W = grid.shape
    x_origin, y_origin = origin

    occ = (grid == occupied_value)

    visited = np.zeros_like(occ, dtype=bool)
    polygons = []

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r0 in range(H):
        for c0 in range(W):
            if not occ[r0, c0] or visited[r0, c0]:
                continue

            # BFS/DFS to find all cells in this connected component
            queue_ = deque()
            queue_.append((r0, c0))
            visited[r0, c0] = True

            min_r = max_r = r0
            min_c = max_c = c0

            while queue_:
                r, c = queue_.pop()
                # update bounding box
                if r < min_r: min_r = r
                if r > max_r: max_r = r
                if c < min_c: min_c = c
                if c > max_c: max_c = c

                for dr, dc in neighbors:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and not visited[rr, cc] and occ[rr, cc]:
                        visited[rr, cc] = True
                        queue_.append((rr, cc))

            x0 = x_origin + min_c * cell_size
            x1 = x_origin + (max_c + 1) * cell_size
            y0 = y_origin + (H - 1 - max_r) * cell_size
            y1 = y_origin + (H - min_r) * cell_size

            vertices = np.array([
                [x0, x1, x1, x0],  # x coordinates
                [y0, y0, y1, y1],  # y coordinates
            ])

            polygons.append(vertices)

    return PolygonWorld(vertices=polygons)

def get_relative_path_waypoints(obs, debug):
    y_limits = (5,15)
    x_limits = (0,20)
    cell_size = 0.1
    origin = (0.0, 0.0)

    grid = obs["grid"]
    grid[grid == 0.5] = 0

    H, W = grid.shape

    # Convert start and goal from grid indices to polygon-world coordinates
    x_start = cell_size*np.array(obs["robot_pos"])
    x_goal  = cell_size*np.array(obs["goal"])
    x_start = TwoDimArray(x_start[0],  (H*cell_size) - x_start[1])
    x_goal  = TwoDimArray(x_goal[0],  (H*cell_size) - x_goal[1])
    polygon_world = grid_to_polygon_world_big_components(grid, cell_size, origin)

    if debug:
        fig, ax = plt.subplots(2, figsize=(10, 10))
        pt.plot_polygon_world(ax=ax[0], world=polygon_world)
        ax[0].scatter(x_start.x, x_start.y, c="red")
        ax[0].scatter(x_goal.x, x_goal.y, c="green")
        ax[1].imshow(grid)
        ax[1].scatter(*obs["robot_pos"], c="red")
        ax[1].scatter(*obs["goal"], c="green")
        plt.show()

    x_vec, y_vec = run_rrt_planner("rrt_star", x_start, x_goal, polygon_world, y_limits, x_limits, False, 100)
    plan = x_vec, y_vec

    return plan, polygon_world, x_start

def get_dwa_sim(plan, obstacle_world, x_start, step_dt):
    """Runs an example of the dynamic window approach with a unicycle dynamic model.
    """

    # Initialize the state and control
    state_initial = UnicycleState(x = x_start.x, y= x_start.y, psi= 0.)

    # Initialize the dwa search parameters
    ds = 0.05
    dwa_params = DwaParams(v_des=2.,
                           w_max=1.,
                           w_res=0.1,
                           ds=ds,
                           sf=2.,
                           s_eps=0.1,
                           k_v=2.,
                           sigma=2.,
                           classic=False,
                           v_res=0.25)

    # Create an inflated grid from the world
    grid = og.generate_occupancy_from_polygon_world(
        world=obstacle_world,
        res=0.1,
        x_lim=(0,20),
        y_lim=(5,15))
    inf_grid = og.inflate_obstacles(grid=grid, inflation=0.25)

    line = np.array([plan[0], plan[1]])
    carrot = LineCarrot(line=line, s_dev_max=5., s_carrot=2.)

    # Create the simulation
    params = SimParameters(initial_state=state_initial)
    params.sim_plot_period = 0.1
    params.sim_step = step_dt
    params.sim_update_period = step_dt
    sim = DwaFollowerIsaacLab(params=params,
                      dynamics= uni.dynamics,
                      controller=uni.arc_control,
                      dynamic_params= uni.UnicycleParams(),
                      dwa_params=dwa_params,
                      n_inputs=UnicycleControl.n_inputs,
                      world=inf_grid,
                      carrot=carrot
                      )
    return sim

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    args = args_cli.__dict__

    args["env"] = "isaaclab"
    args["exp_name"] = "play"

    env_args = {}
    env_cfg.scene.num_envs = args["num_envs"]
    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    env_args["headless"] = args["headless"]
    debug = env_args["debug"] = args["debug"]

    env = gym.make(env_args["task"], cfg=env_cfg, render_mode=None)
    env.reset()
    path_planned = False

    sim = None

    while simulation_app.is_running():
        with torch.inference_mode():
            if not path_planned:
                actions = torch.zeros(1,2)
                obs, _, _, _, _ = env.step({"robot_0":actions})
                obs = obs["robot_0"]
                print("Planning Path")
                plan, obstacle_world, x_start = get_relative_path_waypoints(obs, debug)
                sim = get_dwa_sim(plan, obstacle_world, x_start, env.unwrapped.step_dt)
                path_planned = True
                print("Path Planned")
            else:
                actions = torch.zeros(1,2)
                velocity, steer = sim.dwa_arc.v, sim.dwa_arc.w
                actions[:, 0] = velocity
                actions[:, 0] = steer
                env.step({"robot_0":actions})

            if env.unwrapped.sim._number_of_steps >= args["num_env_steps"]:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
