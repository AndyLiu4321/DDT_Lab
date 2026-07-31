# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NP3O training configs for Tita.

Built on top of [agents/np3o_cfg.py](../../agents/np3o_cfg.py) base; only
``experiment_name`` / ``max_iterations`` differ. Numbers mirror
``LocomotionWithNP3O/configs/tita/tita_flat_config.py``.
"""

from __future__ import annotations

from ddt_lab.tasks.manager_based.locomotion.agents.np3o_cfg import base_np3o_runner_cfg


def tita_flat_np3o_runner_cfg() -> dict:
    cfg = base_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "tita_flat"
    cfg["runner"]["max_iterations"] = 3000
    return cfg


def tita_rough_np3o_runner_cfg() -> dict:
    cfg = base_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "tita_rough"
    cfg["runner"]["max_iterations"] = 5000
    return cfg


def tita_jump_np3o_runner_cfg() -> dict:
    """NP3O runner for the Tita 8DOT jump + recovery task."""
    cfg = base_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "tita_jump"
    cfg["runner"]["max_iterations"] = 20000
    return cfg


def tita_command_gated_flat_np3o_runner_cfg() -> dict:
    cfg = base_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "tita_command_gated_flat"
    cfg["runner"]["num_steps_per_env"] = 25
    cfg["runner"]["max_iterations"] = 30000
    cfg["algorithm"]["learning_rate"] = 1.0e-4
    cfg["algorithm"]["gamma"] = 0.99
    cfg["algorithm"]["lam"] = 0.95
    cfg["algorithm"]["desired_kl"] = 0.01
    cfg["algorithm"]["entropy_coef"] = 0.01
    cfg["policy"]["actor_hidden_dims"] = [512, 512, 256, 128]
    cfg["policy"]["critic_hidden_dims"] = [512, 512, 256, 128]
    cfg["policy"]["activation"] = "elu"
    cfg["policy"]["init_noise_std"] = 1.5
    return cfg
