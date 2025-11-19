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

parser = argparse.ArgumentParser(description="Train an RL agent with HARL.")
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
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"pysim_integration_cfg_entry_point"


import torch
from py_sim.worlds.polygon_world import PolygonWorld
import py_sim.plotting.plotting as pt
import numpy as np
import matplotlib.pyplot as plt
from py_sim.tools.sim_types import TwoDimArray, UnicycleState


def grid_to_polygon_world(grid_tensor, cell_size=1.0, origin=(0.0, 0.0), occupied_value=1):
    """
    Convert a 2D occupancy grid (0/1) into a PolygonWorld with one rectangle per occupied cell.

    grid_tensor: torch.Tensor or np.ndarray of shape (H, W)
    cell_size: length of one cell in world units
    origin: (x0, y0) world coordinate of grid cell (row=0, col=0)
    occupied_value: value in grid that represents an obstacle
    """
    # Convert to numpy if needed
    if isinstance(grid_tensor, torch.Tensor):
        grid = grid_tensor.cpu().numpy()
    else:
        grid = grid_tensor

    H, W = grid.shape
    x_origin, y_origin = origin

    polygons = []

    # Find indices of occupied cells
    rows, cols = np.where(grid == occupied_value)

    for r, c in zip(rows, cols):
        # --- coordinate convention #1: row = y increasing downward ---
        # If you treat row index r as vertical (y) increasing downward:
        x0 = x_origin + c * cell_size
        x1 = x0 + cell_size
        y0 = y_origin + r * cell_size
        y1 = y0 + cell_size

        # If instead you want (row=0) at the TOP and (y) increasing UP,
        # use this instead for y:
        # y0 = y_origin + (H - 1 - r) * cell_size
        # y1 = y0 + cell_size

        # 4-vertex rectangle polygon in the same format as your V1/V2/V3
        vertices = np.array([
            [x0, x1, x1, x0],  # x coordinates
            [y0, y0, y1, y1],  # y coordinates
        ])
        polygons.append(vertices)

    return PolygonWorld(vertices=polygons)

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    args = args_cli.__dict__

    args["env"] = "isaaclab"
    args["algo"] = args["algorithm"]
    args["exp_name"] = "play"

    env_args = {}
    env_cfg.scene.num_envs = args["num_envs"]
    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    env_args["headless"] = args["headless"]
    env_args["debug"] = args["debug"]

    env = gym.make(env_args["task"], cfg=env_cfg, render_mode=None)
    # create runner

    fig, ax = plt.subplots()
    env.reset()

    plot_world = False


    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(1,2)
            actions[:, 0] = 1
            obs, _, _, _, _ = env.step({"robot_0":actions})
            obs = obs["robot_0"]

            if not plot_world:
                grid = obs["grid"]
                grid[grid == 0.5] = 0
                polygon_world = grid_to_polygon_world(grid, 0.1)
                pt.plot_polygon_world(ax=ax, world=polygon_world)
                plot_world = True

            if env.unwrapped.sim._number_of_steps >= args["num_env_steps"]:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
