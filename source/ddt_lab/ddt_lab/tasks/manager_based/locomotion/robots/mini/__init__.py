# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, recovery_env_cfg, rough_env_cfg

##
# Register Gym environments. All Mini tasks are wired to NP3O (BarlowTwins-PPO).
##

gym.register(
    id="DDT-Velocity-Flat-Mini-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MiniFlatEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:mini_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-Mini-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MiniFlatEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:mini_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-Mini-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.MiniRoughEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:mini_rough_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-Mini-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.MiniRoughEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:mini_rough_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Recovery-Rough-Mini-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": recovery_env_cfg.MiniRecoveryRoughEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:mini_rough_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Recovery-Rough-Mini-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": recovery_env_cfg.MiniRecoveryRoughEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:mini_rough_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Recovery-Flat-Mini-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": recovery_env_cfg.MiniRecoveryEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:mini_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Recovery-Flat-Mini-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": recovery_env_cfg.MiniRecoveryEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:mini_flat_np3o_runner_cfg",
    },
)
