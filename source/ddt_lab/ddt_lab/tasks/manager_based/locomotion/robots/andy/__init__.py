# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations for the Andy robot."""

import gymnasium as gym

from . import agents, flat_env_cfg, height_env_cfg, jump_env_cfg, recovery_env_cfg, rough_env_cfg


def _register(task_id: str, env_cfg, runner: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:{runner}",
        },
    )


_register("DDT-Velocity-Flat-Andy-v0", flat_env_cfg.AndyFlatEnvCfg, "andy_flat_np3o_runner_cfg")
_register("DDT-Velocity-Flat-Andy-Play-v0", flat_env_cfg.AndyFlatEnvCfg_PLAY, "andy_flat_np3o_runner_cfg")
_register("DDT-Velocity-Rough-Andy-v0", rough_env_cfg.AndyRoughEnvCfg, "andy_rough_np3o_runner_cfg")
_register("DDT-Velocity-Rough-Andy-Play-v0", rough_env_cfg.AndyRoughEnvCfg_PLAY, "andy_rough_np3o_runner_cfg")
_register("DDT-Recovery-Flat-Andy-v0", recovery_env_cfg.AndyRecoveryEnvCfg, "andy_flat_np3o_runner_cfg")
_register(
    "DDT-Recovery-Flat-Andy-Play-v0", recovery_env_cfg.AndyRecoveryEnvCfg_PLAY, "andy_flat_np3o_runner_cfg"
)
_register("DDT-Recovery-Rough-Andy-v0", recovery_env_cfg.AndyRecoveryRoughEnvCfg, "andy_rough_np3o_runner_cfg")
_register(
    "DDT-Recovery-Rough-Andy-Play-v0",
    recovery_env_cfg.AndyRecoveryRoughEnvCfg_PLAY,
    "andy_rough_np3o_runner_cfg",
)
_register("DDT-jump-Flat-Andy-v0", jump_env_cfg.AndyJumpFlatEnvCfg, "andy_jump_np3o_runner_cfg")
_register(
    "DDT-jump-Flat-Andy-Play-v0",
    jump_env_cfg.AndyJumpFlatEnvCfg_PLAY,
    "andy_jump_np3o_runner_cfg",
)

_register("DDT-Height-Flat-Andy-v0", height_env_cfg.AndyHeightFlatEnvCfg, "andy_height_np3o_runner_cfg")
_register("DDT-Height-Flat-Andy-Play-v0", height_env_cfg.AndyHeightFlatEnvCfg_PLAY, "andy_height_np3o_runner_cfg")

# PACE owns its task registrations and historical import aliases.
from . import pace  # noqa: F401
