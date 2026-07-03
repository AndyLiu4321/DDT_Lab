# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mini flat-terrain full-flow recovery + jump training env.

Stage-based jump reward design:

    jump_cmd is sampled per reset/segment: P(jump_cmd=1)=0.2,
    P(jump_cmd=0)=0.8. The no-jump samples keep recovery behavior alive while
    jump samples enter the stage machine.

  Stage PREP (pre-trigger)   — jump_before_setting (+2.0)
    Robot crouches before launching while jump_cmd=1.

  Stage TAKEOFF/FLIGHT — lin_vel_z_jump (+5.0)
    Reward upward world-z velocity after the command is issued.

  Stage FLIGHT short window — jump_flight_height (+14.0)
    Reward current COM height only during a short in-flight window.

  Stage 3 (landing recovery) — jump_land_stability (+5.0) + jump_land_orientation (+3.0)
    After landing: reward low angular velocity and upright posture.

Jump command:
  Implemented by mdp.JumpCommandCfg, following wheelfoot_flat_jump's FSM:
  trigger frame 125, S1 timeout 5 s, S3 recovery timeout 3 s, then retrigger.
  No z-velocity assist or upward push is applied.

Train directly:
  python scripts/np3o/train.py \\
    --task DDT-jump-Flat-Mini-v0 \\
    --num_envs 4096 \\
    --max_iterations 20000
"""

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .flat_env_cfg import MiniFlatEnvCfg



@configclass
class MiniJumpFlatEnvCfg(MiniFlatEnvCfg):
    """Mini flat-terrain jump training environment."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 15.0
        self.terminations.base_contact = None

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
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

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

        # ------------------------------------------------------------------ #
        # Full-flow initialization: recovery warmup, then jump commands       #
        # ------------------------------------------------------------------ #
        # JumpCommand samples mostly no-jump segments so recovery remains in
        # distribution while jump segments follow:
        # prep -> takeoff -> short flight reward window -> landing recovery.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.05, 0.5),
                "roll": (-3.14, 3.14),
                "pitch": (-1.5708, 1.5708),
                "yaw":   (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll":  (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw":   (-0.5, 0.5),
            },
        }

        self.events.jump_assist = None

        # ------------------------------------------------------------------ #
        # Recovery warmup + jump rewards                                      #
        # ------------------------------------------------------------------ #

        self.rewards.base_contact_raw.weight = -2.0
        self.rewards.base_contact_penalty.weight = 0.0
        self.rewards.upright_progress.weight = 2.0
        self.rewards.inverted_ang_vel_bonus.weight = 0.1
        self.rewards.base_height_l2.weight = -10.0

        # Z-velocity penalty — directly opposes takeoff
        self.rewards.lin_vel_z_l2.weight = 0.0

        # XY angular velocity penalty — in-flight rotation is acceptable
        self.rewards.ang_vel_xy_l2.weight = 0.0

        # Keep the policy responsive to the same [vx, vy, wz] command used by
        # keyboard play, without letting locomotion dominate jump learning.
        self.rewards.track_lin_vel_xy_exp.weight = 0.6
        self.rewards.track_ang_vel_z_exp.weight = 0.3

        # Orientation penalty: reduce (not zero) — allow tilting in flight
        # but still penalise extreme roll/pitch on the ground
        self.rewards.flat_orientation_l2.weight = -5.0

        # ------------------------------------------------------------------ #
        # General uprightness — keeps robot standing when not airborne        #
        # ------------------------------------------------------------------ #
        self.rewards.upward.weight = 1.0

        # ------------------------------------------------------------------ #
        # Stage 0: Pre-jump crouch (蓄力)                                    #
        # ------------------------------------------------------------------ #
        # Gaussian reward centred at crouch_height=0.25 m (below standing 0.35).
        # Fires only when at least one foot is on the ground.
        self.rewards.jump_before_setting.weight = 2.0
        self.rewards.jump_before_setting.params["command_name"] = "jump_cmd"

        # ------------------------------------------------------------------ #
        # Stage 1: Upward z-velocity — fires on ground AND in air             #
        # ------------------------------------------------------------------ #
        # KEY FIX: require_in_air=False provides gradient toward takeoff even
        # while still on the ground. Without this, the reward is completely
        # sparse until the robot accidentally discovers jumping.
        self.rewards.lin_vel_z_jump.weight = 18.0
        self.rewards.lin_vel_z_jump.params["require_in_air"] = False
        self.rewards.lin_vel_z_jump.params["command_name"] = "jump_cmd"

        # ------------------------------------------------------------------ #
        # Stage 2: Flight — reward height above ground                       #
        # ------------------------------------------------------------------ #
        # Dominant signal, but gated to the early flight window only.
        self.rewards.jump_flight_height.weight = 28.0
        self.rewards.jump_flight_height.params["command_name"] = "jump_cmd"
        self.rewards.jump_flight_height.params["target_height"] = 0.8
        self.rewards.jump_flight_height.params["min_height"] = 0.42

        # ------------------------------------------------------------------ #
        # Stage 3: Landing — stability + orientation                         #
        # ------------------------------------------------------------------ #
        # Penalise angular velocity on landing (tumbling → negative reward).
        self.rewards.jump_land_stability.weight = 5.0
        self.rewards.jump_land_stability.params["command_name"] = "jump_cmd"

        # Reward upright orientation (1-g_z)^2 when on ground.
        self.rewards.jump_land_orientation.weight = 3.0
        self.rewards.jump_land_orientation.params["command_name"] = "jump_cmd"


# @configclass
# class MiniJumpFlatEnvCfg_PLAY(MiniJumpFlatEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         self.observations.policy.enable_corruption = False
#         self.events.base_external_force_torque = None
#         self.events.push_robot = None
#         self.events.jump_assist = None   # disable z-velocity assist in play mode
#         self.events.add_base_inertia = None
#         self.events.add_base_com = None
#         self.events.add_base_mass = None
#         self.events.randomize_actuator_gains = None
#         self.commands.base_velocity.debug_vis = True
#         self.commands.jump.play_mode = True
#         self.commands.jump.play_trigger_steps = 250
#         self.commands.jump.warmup_iterations = 0
#         # Reset to ground-only for play (no flight-phase init)
#         self.events.reset_base.params["pose_range"]["z"] = (0.35, 0.35)
#         self.events.reset_base.params["pose_range"]["roll"] = (0.0, 0.0)
#         self.events.reset_base.params["pose_range"]["pitch"] = (0.0, 0.0)
#         self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
#         self.events.reset_base.params["velocity_range"]["x"] = (0.0, 0.0)
#         self.events.reset_base.params["velocity_range"]["y"] = (0.0, 0.0)
#         self.events.reset_base.params["velocity_range"]["z"] = (0.0, 0.0)
#         self.events.reset_base.params["velocity_range"]["roll"] = (0.0, 0.0)
#         self.events.reset_base.params["velocity_range"]["pitch"] = (0.0, 0.0)
#         self.events.reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)

class MiniJumpFlatEnvCfg_PLAY(MiniJumpFlatEnvCfg):
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

        # play 模式下先完全关闭随机速度命令
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1e9, 1e9)

        self.commands.jump_cmd.play_mode = True
        self.commands.jump_cmd.play_trigger_steps = 250
        self.commands.jump_cmd.warmup_iterations = 0

        self.events.reset_base.params["pose_range"]["z"] = (0.35, 0.35)
        self.events.reset_base.params["pose_range"]["roll"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["roll"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)
