# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NP3O runner configurations for Andy."""

from ddt_lab.tasks.manager_based.locomotion.agents.np3o_cfg import base_np3o_runner_cfg


def andy_flat_np3o_runner_cfg() -> dict:
    cfg = base_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "andy_flat"
    cfg["runner"]["max_iterations"] = 3000
    return cfg


def andy_rough_np3o_runner_cfg() -> dict:
    cfg = base_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "andy_rough"
    cfg["runner"]["max_iterations"] = 5000
    return cfg


def andy_jump_np3o_runner_cfg() -> dict:
    cfg = base_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "andy_jump"
    cfg["runner"]["max_iterations"] = 20000
    return cfg


def andy_height_np3o_runner_cfg() -> dict:
    """Use the existing Andy flat NP3O training pipeline."""
    cfg = andy_flat_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "andy_height"
    cfg["runner"]["max_iterations"] = 20000
    return cfg
