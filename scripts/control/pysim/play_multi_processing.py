# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run Isaac Lab with a one-shot RRT planner in a separate process."""

from __future__ import annotations

import argparse
import sys
import multiprocessing as mp
import queue  # for Empty

import torch
from isaaclab.app import AppLauncher
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# CLI / AppLauncher (must be BEFORE isaaclab imports)
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train an RL agent with HARL + one-shot RRT planning.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="happo",
    choices=[
        "happo",
        "hatrpo",
        "haa2c",
        "haddpg",
        "hatd3",
        "hasac",
        "had3qn",
        "maddpg",
        "matd3",
        "mappo",
        "happo_adv",
    ],
    help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_env_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--dir", type=str, default=None, help="folder with trained models")
parser.add_argument("--debug", action="store_true", help="whether to run in debug mode for visualization")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments once
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app (must be before isaaclab imports)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------------------
# Now it's safe to import Isaac Lab / Hydra stuff
# --------------------------------------------------------------------------------------

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from py_sim.worlds.polygon_world import PolygonWorld  # noqa: E402
import py_sim.plotting.plotting as pt  # noqa: E402
from py_sim.tools.sim_types import TwoDimArray, UnicycleState  # noqa: E402
from py_sim.isaaclab.sample_planning import run_rrt_planner  # noqa: E402

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "pysim_integration_cfg_entry_point"

DEBUG = False

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def grid_to_polygon_world(grid_tensor, cell_size=1.0, origin=(0.0, 0.0), occupied_value=1):
    """
    Convert a 2D occupancy grid (0/1) into a PolygonWorld with one rectangle per occupied cell.
    """
    if isinstance(grid_tensor, torch.Tensor):
        grid = grid_tensor.cpu().numpy()
    else:
        grid = np.array(grid_tensor)

    H, W = grid.shape
    x_origin, y_origin = origin

    polygons = []

    rows, cols = np.where(grid == occupied_value)

    for r, c in zip(rows, cols):
        x0 = x_origin + c * cell_size
        x1 = x0 + cell_size

        # flip y so row 0 is top
        y0 = y_origin + (H - 1 - r) * cell_size
        y1 = y0 + cell_size

        vertices = np.array(
            [
                [x0, x1, x1, x0],  # x coords
                [y0, y0, y1, y1],  # y coords
            ]
        )
        polygons.append(vertices)

    return PolygonWorld(vertices=polygons)


# --------------------------------------------------------------------------------------
# Planner process – runs ONCE on a single observation
# --------------------------------------------------------------------------------------

def planner_process(obs_msg: dict, path_queue: mp.Queue, debug: bool = False):
    """
    One-shot planner: takes a single observation dict, computes a path, writes it to path_queue.
    Runs in a separate process.
    """
    grid = np.array(obs_msg["grid"], copy=True)
    robot_pos = np.array(obs_msg["robot_pos"], dtype=float)
    goal = np.array(obs_msg["goal"], dtype=float)

    # clean grid: treat 0.5 as free
    grid[grid == 0.5] = 0

    H, W = grid.shape
    cell_size = 0.1
    origin = (0.0, 0.0)
    y_limits = (5, 15)
    x_limits = (0, 20)

    # Convert to world coords
    x_start_arr = cell_size * robot_pos
    x_goal_arr = cell_size * goal

    x_start = TwoDimArray(x_start_arr[0], (H * cell_size) - x_start_arr[1])
    x_goal = TwoDimArray(x_goal_arr[0], (H * cell_size) - x_goal_arr[1])

    polygon_world = grid_to_polygon_world(grid, cell_size, origin)

    if debug:
        fig, ax = plt.subplots(2, figsize=(10, 10))
        pt.plot_polygon_world(ax=ax[0], world=polygon_world)
        ax[0].scatter(x_start.x, x_start.y, c="red")
        ax[0].scatter(x_goal.x, x_goal.y, c="green")
        ax[1].imshow(grid)
        ax[1].scatter(robot_pos[0], robot_pos[1], c="red")
        ax[1].scatter(goal[0], goal[1], c="green")
        plt.show(block=False)

    # Heavy RRT* call – done in this separate process
    x_vec, y_vec = run_rrt_planner(
        "rrt_star",
        x_start,
        x_goal,
        polygon_world,
        y_limits,
        x_limits,
        False,
        100,
    )

    # Send the path back to the main process
    path_queue.put((x_vec, y_vec))


# --------------------------------------------------------------------------------------
# Main Hydra entrypoint
# --------------------------------------------------------------------------------------

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    global DEBUG

    args = args_cli.__dict__
    args["env"] = "isaaclab"
    args["algo"] = args["algorithm"]
    args["exp_name"] = "play"

    env_args: dict = {}
    env_cfg.scene.num_envs = args["num_envs"]
    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    env_args["headless"] = args["headless"]
    DEBUG = env_args["debug"] = args["debug"]

    # Create Isaac Lab env
    env = gym.make(env_args["task"], cfg=env_cfg, render_mode=None)
    env.reset()

    # One-shot planner state
    path_queue: mp.Queue = mp.Queue(maxsize=1)
    planner_proc: mp.Process | None = None
    planner_started = False
    path_planned = False
    current_path = None

    while simulation_app.is_running():
        with torch.inference_mode():
            # TODO: once path is available, replace this with real path-following logic
            actions = torch.zeros(1, 2)

            obs, _, _, _, _ = env.step({"robot_0": actions})
            obs_robot = obs["robot_0"]

            # Start planner once, using the first observation we care about
            if not planner_started:
                grid = obs_robot["grid"]
                robot_pos = obs_robot["robot_pos"]
                goal = obs_robot["goal"]

                # Convert to numpy / CPU for pickling to the child process
                if isinstance(grid, torch.Tensor):
                    grid = grid.cpu().numpy()
                if isinstance(robot_pos, torch.Tensor):
                    robot_pos = robot_pos.cpu().numpy()
                if isinstance(goal, torch.Tensor):
                    goal = goal.cpu().numpy()

                obs_msg = {
                    "grid": grid,
                    "robot_pos": robot_pos,
                    "goal": goal,
                }

                planner_proc = mp.Process(
                    target=planner_process,
                    args=(obs_msg, path_queue, DEBUG),
                    daemon=True,
                )
                planner_proc.start()
                planner_started = True

            # Try to non-blockingly get the path (once) from the planner
            if planner_started and not path_planned:
                try:
                    x_vec, y_vec = path_queue.get_nowait()
                    current_path = (x_vec, y_vec)
                    path_planned = True
                    # At this point you can change `actions` above to follow `current_path`.
                    # e.g., compute a velocity command toward the next waypoint.
                    print(f"Planner finished, got path with {len(x_vec)} points.")
                except queue.Empty:
                    pass

            # Stop condition
            if env.unwrapped.sim._number_of_steps >= args["num_env_steps"]:
                break

    # Cleanup
    env.close()

    if planner_proc is not None:
        planner_proc.join(timeout=1.0)


# --------------------------------------------------------------------------------------
# Script entrypoint
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
