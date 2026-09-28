# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tita 8DOT flat-terrain recovery and command-triggered jump task.

The jump command FSM and every recovery/jump reward use the same functions and
weights as ``robots/mini/jump_env_cfg.py``. The robot-specific action and joint
observations are inherited from :class:`TitaFlatEnvCfg`, so Tita keeps all eight
actions instead of copying Mini's six-action layout.

Expected observation/action sizes:

* policy: 34 = Tita flat 33 + ``jump_cmd`` 1;
* critic: 59 = Tita flat 54 + ``jump_cmd`` 1 + ``jump_state`` 4;
* action: 8 = left/right ``leg_1/2/3/4``.
"""

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .flat_env_cfg import TitaFlatEnvCfg


_RECOVERY_LEG_2_LIMITS = (-1.5, 2.0)


@configclass
class TitaJumpFlatEnvCfg(TitaFlatEnvCfg):
    """Tita 8DOT joint recovery and staged-jump training environment."""

    def __post_init__(self):
        super().__post_init__()

        # Tita inherits the same 0.02 s control period as Mini. A 15 s episode
        # therefore contains at most 750 policy steps.
        self.episode_length_s = 15.0
        # Base contact is penalized continuously but does not terminate the
        # episode, giving the 8DOT robot time to learn self-righting.
        self.terminations.base_contact = None

        # Shared Mini jump FSM. Both robots use left/right leg_4 wheel contact
        # to detect TAKEOFF, FLIGHT and LAND; this part is independent of DOF.
        self.commands.jump_cmd = mdp.JumpCommandCfg(
            trigger_step_range=(125, 126),
            warmup_iterations=500,
            steps_per_iteration=24,
            target_height=0.8,
            jump_probability=0.2,
            recovery_resample_s=3.0,
            flight_reward_window_s=0.8,
            sensor_cfg=SceneEntityCfg("contact_forces", body_names=[".*_leg_4"]),
        )

        # Keep Tita's native [vx, vy, wz] velocity command while reducing the
        # range so locomotion does not dominate early jump learning.
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # JointPositionAction 的 clip 是缩放后的实际位置目标（单位 rad）。
        # 禁止 leg_2 把目标送到 URDF 虽允许、但会跨到机身反面的约 3 rad；
        # Tita 其余 6 个关节仍保留原来的 8DOT 动作范围用于摆腿恢复。
        self.actions.joint_pos_1.clip = {".*": _RECOVERY_LEG_2_LIMITS}
        self.actions.joint_pos_5.clip = {".*": _RECOVERY_LEG_2_LIMITS}

        # Actor gets only jump intent. Critic additionally gets the privileged
        # FSM state: stage, was_in_flight, has_jumped and normalized max height.
        self.observations.policy.jump_cmd = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "jump_cmd"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.jump_cmd = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "jump_cmd"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.jump_state = ObsTerm(
            func=mdp.jump_state,
            params={"command_name": "jump_cmd"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # ``reset_root_state_uniform`` adds xyz ranges to the asset's default
        # root pose. Tita's default z is 0.40 m, so (-0.35, 0.10) produces the
        # intended absolute 0.05--0.50 m recovery range used by Mini's design.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.35, 0.10),
                "roll": (-3.14, 3.14),
                "pitch": (-1.5708, 1.5708),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        # 只在 Tita Jump 任务中收紧 PhysX 物理限位。原 URDF 上限为
        # 3.490659 rad，原有 joint_pos_limit cost 因而不会惩罚 3 rad。
        # 写入新硬限位后，Isaac Lab 也会重算 NP3O cost 使用的 soft limit。
        self.events.limit_recovery_leg_2 = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["joint_left_leg_2", "joint_right_leg_2"]
                ),
                "lower_limit_distribution_params": (
                    _RECOVERY_LEG_2_LIMITS[0],
                    _RECOVERY_LEG_2_LIMITS[0],
                ),
                "upper_limit_distribution_params": (
                    _RECOVERY_LEG_2_LIMITS[1],
                    _RECOVERY_LEG_2_LIMITS[1],
                ),
                "operation": "abs",
                "distribution": "uniform",
            },
        )

        # Recovery/stability budget copied from Mini jump.
        self.rewards.base_contact_raw.weight = -2.0
        self.rewards.base_contact_penalty.weight = 0.0
        self.rewards.upright_progress.weight = 2.0
        # 不再奖励倒置时任意转动机身，改为奖励腿部相对 base 产生的净角动量。
        # upright_progress 仍负责保证摆腿最终必须让机身朝直立方向恢复。
        self.rewards.inverted_ang_vel_bonus.weight = 0.0
        self.rewards.inverted_leg_swing_momentum.weight = 0.5
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.6
        self.rewards.track_ang_vel_z_exp.weight = 0.3
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.upward.weight = 1.0

        # PREP: crouch before the fixed trigger step.
        self.rewards.jump_before_setting.weight = 2.0
        self.rewards.jump_before_setting.params["command_name"] = "jump_cmd"

        # TAKEOFF/FLIGHT: provide a dense positive-z velocity signal before and
        # after wheel contact is lost.
        self.rewards.lin_vel_z_jump.weight = 18.0
        self.rewards.lin_vel_z_jump.params["require_in_air"] = False
        self.rewards.lin_vel_z_jump.params["command_name"] = "jump_cmd"

        # FLIGHT: reward height progress only in the short early-flight window.
        self.rewards.jump_flight_height.weight = 28.0
        self.rewards.jump_flight_height.params["command_name"] = "jump_cmd"
        self.rewards.jump_flight_height.params["target_height"] = 0.8
        self.rewards.jump_flight_height.params["min_height"] = 0.42

        # LAND: positive exponential rewards for low roll/pitch rate and an
        # upright body orientation after wheel contact returns.
        self.rewards.jump_land_stability.weight = 5.0
        self.rewards.jump_land_stability.params["command_name"] = "jump_cmd"
        self.rewards.jump_land_orientation.weight = 3.0
        self.rewards.jump_land_orientation.params["command_name"] = "jump_cmd"


@configclass
class TitaJumpFlatEnvCfg_PLAY(TitaJumpFlatEnvCfg):
    """Deterministic Tita 8DOT playback configuration."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # Disable training-only disturbances and dynamics randomization.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None

        # Default playback is stationary. ``play.py --keyboard`` overwrites
        # these commands through Tita's native command observation path.
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

        self.commands.jump_cmd.play_mode = True
        self.commands.jump_cmd.play_trigger_steps = 250
        self.commands.jump_cmd.warmup_iterations = 0

        # Position ranges are offsets from Tita's default root pose, so zero z
        # places the robot at its configured 0.40 m standing height.
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["roll"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["roll"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)
