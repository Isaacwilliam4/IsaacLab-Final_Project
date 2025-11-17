# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Leatherback Rover locomotion environment.
"""

import gymnasium as gym

from . import agents

from .leatherback_path_planning import LeatherbackPathPlanningEnv, LeatherbackPathPlanningEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="leatherback-path-planning-v0",
    entry_point=LeatherbackPathPlanningEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackPathPlanningEnvCfg,
        "pysim_integration_cfg_entry_point": f"{agents.__name__}:pysim_integration_cfg.yaml",
    },
)