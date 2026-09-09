# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Andy fall-recovery configurations for flat and rough terrain."""

import math

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ddt_lab.tasks.manager_based.locomotion.robots.mini.recovery_env_cfg import (
    MiniRecoveryEnvCfg,
    MiniRecoveryRoughEnvCfg,
)

from .rough_env_cfg import (
    ANDY_AUXILIARY_WHEEL_BODIES,
    ANDY_POSITION_JOINTS,
    AndyActionsCfg,
    AndySceneCfg,
    configure_andy_mdp,
)


ANDY_RECOVERY_JOINT_POS = {
    "joint_left_leg_1": 0.0,
    "joint_left_leg_2": math.radians(40.0),
    "joint_left_leg_3": 0.0,
    "joint_right_leg_1": 0.0,
    "joint_right_leg_2": math.radians(40.0),
    "joint_right_leg_3": 0.0,
}


def configure_andy_domain_randomization(env_cfg) -> None:
    """Reduce Mini's domain-randomization deviations for the lighter Andy."""

    material_params = env_cfg.events.physics_material.params
    # Preserve the original 0.9 friction midpoint and reduce its span to 1/4.
    material_params["static_friction_range"] = (0.725, 1.075)
    material_params["dynamic_friction_range"] = (0.725, 1.075)
    # Restitution is centered at the physical nominal value of zero; keeping
    # values near one would still allow highly elastic ground impacts.
    material_params["restitution_range"] = (0.0, 0.25)

    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (-0.125, 0.5)
    env_cfg.events.add_base_inertia.params["inertia_distribution_params"] = (0.975, 1.025)
    env_cfg.events.add_base_com.params["com_range"] = {
        "x": (-0.0125, 0.0125),
        "y": (-0.0125, 0.0125),
        "z": (-0.0125, 0.0125),
    }
    gain_params = env_cfg.events.randomize_actuator_gains.params
    gain_params["stiffness_distribution_params"] = (0.95, 1.05)
    gain_params["damping_distribution_params"] = (0.95, 1.05)


def configure_andy_recovery_rewards(env_cfg) -> None:
    """Shape Andy recovery toward leg-assisted righting and active-wheel support."""

    # Position actions use the articulation's default joint state as their
    # offset. This makes zero action represent the requested recovered pose.
    env_cfg.scene.robot.init_state.joint_pos.update(ANDY_RECOVERY_JOINT_POS)

    # Enforce the recovered pose only while the base is near upright. The MDP
    # term contains an uprightness gate, so inverted leg motion remains free.
    env_cfg.rewards.recovery_joint_pose = RewTerm(
        func=mdp.default_joint_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ANDY_POSITION_JOINTS, preserve_order=True
            )
        },
    )

    # A stronger orientation-independent penalty discourages policies that
    # accept a small population of permanently base-down environments.
    env_cfg.rewards.base_contact_raw.weight = -3.0

    # Arbitrary base angular speed can be earned by rocking or bouncing without
    # producing a useful leg-ground impulse. Reward relative leg momentum while
    # inverted and in base contact instead. Andy is roughly 1/4 the mass of the
    # Tita reference, hence the smaller momentum normalization scale.
    env_cfg.rewards.inverted_ang_vel_bonus.weight = 0.0
    env_cfg.rewards.inverted_leg_swing_momentum = RewTerm(
        func=mdp.inverted_leg_swing_momentum,
        weight=0.5,
        params={
            "momentum_scale": 0.05,
            "contact_force_threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_leg_(1|2|3)"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
        },
    )

    # Use the upright-gated contact term: auxiliary wheels remain available
    # during a fall but become costly once the robot is recovering/upright.
    env_cfg.rewards.auxiliary_wheel_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=ANDY_AUXILIARY_WHEEL_BODIES
            ),
            "threshold": 1.0,
        },
    )


@configclass
class AndyRecoveryEnvCfg(MiniRecoveryEnvCfg):
    """Flat-terrain fall recovery."""

    scene: AndySceneCfg = AndySceneCfg(num_envs=4096, env_spacing=2.5)
    actions: AndyActionsCfg = AndyActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_andy_mdp(self)
        configure_andy_domain_randomization(self)
        configure_andy_recovery_rewards(self)

        # Mini's reset wrench is persistent rather than a one-step impulse.
        # Its +/-10 N and +/-10 N*m ranges are too aggressive for Andy's much
        # lower mass/inertia and can launch the robot after ground contact.
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AndyRecoveryEnvCfg_PLAY(AndyRecoveryEnvCfg):
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


@configclass
class AndyRecoveryRoughEnvCfg(MiniRecoveryRoughEnvCfg):
    """Rough-terrain fall recovery."""

    scene: AndySceneCfg = AndySceneCfg(num_envs=4096, env_spacing=2.5)
    actions: AndyActionsCfg = AndyActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_andy_mdp(self)
        configure_andy_domain_randomization(self)
        configure_andy_recovery_rewards(self)

        # AndyRecoveryRoughEnvCfg inherits MiniRecoveryRoughEnvCfg directly,
        # so repeat AndyRoughEnvCfg's disturbance overrides for training too.
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AndyRecoveryRoughEnvCfg_PLAY(AndyRecoveryRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
