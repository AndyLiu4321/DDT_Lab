# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Andy flat-ground locomotion configuration."""

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import ANDY_AUXILIARY_WHEEL_BODIES, AndyRoughEnvCfg


@configclass
class AndyFlatEnvCfg(AndyRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.upward.weight = 1.0
        self.rewards.upright_progress.weight = 2.0
        self.rewards.inverted_ang_vel_bonus.weight = 0.0
        self.rewards.base_contact_raw.weight = 0.0
        self.rewards.base_contact_penalty.weight = 0.0
        self.rewards.auxiliary_wheel_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ANDY_AUXILIARY_WHEEL_BODIES),
                "threshold": 1.0,
            },
        )

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.scanner = None
        self.curriculum.terrain_levels = None


@configclass
class AndyFlatEnvCfg_PLAY(AndyFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
