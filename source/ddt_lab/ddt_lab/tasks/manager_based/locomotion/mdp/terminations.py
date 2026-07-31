# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat


def roll_pitch_exceeded(env, roll_limit: float, pitch_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Terminate when absolute base roll or pitch exceeds the configured radian limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w, wrap_to_2pi=False)
    return (torch.abs(roll) > roll_limit) | (torch.abs(pitch) > pitch_limit)
