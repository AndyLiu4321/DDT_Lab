# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Andy flat-ground commanded-jump training configuration."""

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ddt_lab.tasks.manager_based.locomotion.robots.mini.jump_env_cfg import MiniJumpFlatEnvCfg

from .recovery_env_cfg import configure_andy_domain_randomization, configure_andy_recovery_rewards
from .rough_env_cfg import (
    ANDY_AUXILIARY_WHEEL_BODIES,
    ANDY_DRIVE_BODIES,
    AndyActionsCfg,
    AndySceneCfg,
    configure_andy_mdp,
)


@configclass
class AndyJumpFlatEnvCfg(MiniJumpFlatEnvCfg):
    """Mini's recovery+jump curriculum retargeted to Andy's six motors."""

    scene: AndySceneCfg = AndySceneCfg(num_envs=4096, env_spacing=2.5)
    actions: AndyActionsCfg = AndyActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Retarget Mini's leg_4 wheel selectors to Andy's driven leg_3 wheels.
        configure_andy_mdp(self)
        self.commands.jump_cmd.sensor_cfg = SceneEntityCfg(
            "contact_forces", body_names=ANDY_DRIVE_BODIES
        )
        self.commands.jump_cmd.warmup_iterations = 50
        self.commands.jump_cmd.jump_probability = 0.4
        self.commands.jump_cmd.takeoff_min_height = 0.36
        self.commands.jump_cmd.takeoff_min_vel_z = 0.05
        self.rewards.jump_flight_height.params["base_height"] = 0.32
        self.rewards.jump_flight_height.params["min_height"] = 0.36
        # Andy needs a deeper PREP crouch than Mini before extending its legs.
        self.rewards.jump_before_setting.weight = 3.0
        self.rewards.jump_before_setting.params["crouch_height"] = 0.23
        self.rewards.jump_before_setting.params["sigma"] = 0.06
        self.rewards.lin_vel_z_jump.weight = 24.0

        # End and reset the environment as soon as the base or either passive
        # support wheel touches the ground.  The inherited Mini jump task
        # disables this termination to permit self-recovery, so explicitly
        # restore it here with Andy's rigid-body names.
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["base_link", *ANDY_AUXILIARY_WHEEL_BODIES],
                ),
                "threshold": 1.0,
            },
        )

        # Start every training episode from Andy's nominal upright state.  All
        # pose values below are offsets from the articulation default state, so
        # a zero z offset retains the configured 0.35 m base height.
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            self.events.reset_base.params["pose_range"][axis] = (0.0, 0.0)
            self.events.reset_base.params["velocity_range"][axis] = (0.0, 0.0)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # Reuse Andy's lighter-robot randomization and recovery shaping. The
        # recovered-pose penalty is disabled here because it would oppose the
        # PREP-stage crouch required for jumping.
        configure_andy_domain_randomization(self)
        configure_andy_recovery_rewards(self)
        self.rewards.recovery_joint_pose = None

        # Do not reward nominal standing height/orientation while the policy is
        # crouching, taking off, or flying.  These terms resume in RECOVERY and
        # LAND so non-jump commands and stable landings remain trained.
        for name in ("base_height_l2", "upward", "upright_progress"):
            getattr(self.rewards, name).params["command_name"] = "jump_cmd"

        # The inherited Mini reset wrench remains active until overwritten and
        # is too strong for Andy. Jumping must come only from commanded motors.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.jump_assist = None


@configclass
class AndyJumpFlatEnvCfg_PLAY(AndyJumpFlatEnvCfg):
    """Deterministic Andy jump playback configuration."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.jump_assist = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

        self.commands.jump_cmd.play_mode = True
        self.commands.jump_cmd.play_trigger_steps = 250
        self.commands.jump_cmd.warmup_iterations = 0

        # reset_root_state_uniform uses offsets relative to Andy's default root
        # pose, so a zero z offset starts at the configured 0.34 m height.
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            self.events.reset_base.params["pose_range"][axis] = (0.0, 0.0)
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            self.events.reset_base.params["velocity_range"][axis] = (0.0, 0.0)
