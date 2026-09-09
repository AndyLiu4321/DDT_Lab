# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mini 地形感知轮腿混合爬楼梯任务。

保留旧类名和 Gym ID 以兼容启动命令，但本任务已经完全移除跳跃FSM、
跳跃奖励和跳跃控制维度：

* 平地：腿关节保持默认支撑姿态，由两个轮速动作完成速度跟踪；
* 不平地形：根据深度地形剖面自动释放腿动作，采用左右交替步态；
* Actor：27-D 本体 + 9-D 深度地形 + 2-D 步态相位，共 38-D；
* RayCaster：只供 Critic、课程和训练期地形门控奖励使用。

轮子在不平地形仍可辅助向前滚动，因此得到的是连续轮腿混合，而不是
必须人工切换的离散“轮式/步态”模式。
"""

import isaaclab.sim as sim_utils
import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .stairs_env_cfg import MiniStairsEnvCfg


_FEET_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=[".*_leg_4"])


@configclass
class MiniStairsJumpEnvCfg(MiniStairsEnvCfg):
    """使用深度地形感知自动混合轮式和交替步态的 Mini 环境。"""

    def __post_init__(self):
        super().__post_init__()

        # 20 s 回合、50 Hz policy/control frequency inherited from stairs:
        # sim.dt=0.005, decimation=4 -> step_dt=0.02 s.
        self.episode_length_s = 20.0
        self.terminations.base_contact = None
        # Real and simulated depth perception both run at 20 Hz.  The 50 Hz
        # policy reuses the latest frame between camera updates.
        self.scene.height_scanner.update_period = 0.05
        self.sim.render_interval = 10

        # Low-resolution, wide-FOV depth camera rigidly mounted on base_link.
        # Positive rotation about +Y pitches the world-convention +X optical
        # axis downward by 35 degrees.
        self.scene.depth_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/depth_camera",
            offset=TiledCameraCfg.OffsetCfg(
                # The base collision box extends to x=0.235 m.  Mount just
                # ahead of it so the camera cannot render the robot shell.
                pos=(0.27, 0.0, 0.04),
                rot=(0.953717, 0.0, 0.300706, 0.0),
                convention="world",
            ),
            update_period=0.05,
            update_latest_camera_pose=False,
            data_types=["depth"],
            depth_clipping_behavior="none",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=1.0,
                horizontal_aperture=20.955,
                clipping_range=(0.08, 3.0),
            ),
            width=48,
            height=32,
        )

        # There is no extra mode command or hidden FSM.  Terrain perception
        # directly determines whether the policy rolls or starts stepping.
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.50)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.25, 0.25)
        self.commands.base_velocity.ranges.heading = (-0.05, 0.05)

        # A failed episode must make less than 0.70 m progress before it is
        # downgraded.  The 0.70--1.40 m neutral band prevents repeated
        # downgrades when a climb ends just short of the upgrade threshold.
        self.curriculum.terrain_levels.params["success_distance"] = 1.40
        self.curriculum.terrain_levels.params["failure_distance"] = 0.70

        # ------------------------------------------------------------------ #
        # Observations: Actor 38-D; raw scanner remains Critic-only           #
        # ------------------------------------------------------------------ #
        # Nine relative heights at x=[0.0, 0.1, ..., 0.8] m, generated from
        # the actual depth image.  No scanner value enters the Actor.
        self.observations.policy.terrain_profile = ObsTerm(
            func=mdp.depth_camera_height_profile,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "asset_cfg": SceneEntityCfg("robot"),
                "mount_position_b": (0.27, 0.0, 0.04),
                "mount_orientation_b": (0.953717, 0.0, 0.300706, 0.0),
                "min_depth": 0.08,
                "max_depth": 3.0,
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-0.20, 0.40),
            scale=1.0,
        )
        # Deployable oscillator used only when the depth profile reports rough
        # terrain.  On flat ground the policy learns to ignore it.
        self.observations.policy.gait_phase = ObsTerm(
            func=mdp.phase,
            params={"cycle_time": 0.8},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.critic.gait_phase = ObsTerm(
            func=mdp.phase,
            params={"cycle_time": 0.8},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Spawn on the central low platform, facing +X.  The first stair edge
        # is about 1 m from the terrain origin, leaving an approach run-up.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (0.20, 0.45),
                "y": (-0.10, 0.10),
                "z": (0.0, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (0.0, 0.10),
                "y": (-0.03, 0.03),
                "z": (-0.03, 0.03),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        }
        # ------------------------------------------------------------------ #
        # Terrain-aware wheel/leg reward balance                              #
        # ------------------------------------------------------------------ #
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        # Strong vertical and double-airborne penalties remove the incentive
        # to jump while still allowing one wheel-foot to step at a time.
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.base_height_l2.weight = -3.0
        self.rewards.joint_mirror.weight = 0.0

        # The gate ramps from wheel mode to gait mode for upcoming terrain
        # height changes between 2.5 cm and 8 cm.
        terrain_gate_params = {
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "low_height": 0.025,
            "high_height": 0.08,
            "forward_start_index": 2,
        }
        self.rewards.flat_leg_posture = RewTerm(
            func=mdp.terrain_gated_flat_leg_posture,
            weight=-2.0,
            params={
                **terrain_gate_params,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_(2|3)"]),
            },
        )
        self.rewards.rough_single_stance = RewTerm(
            func=mdp.terrain_gated_feet_air_time_positive_biped,
            weight=2.0,
            params={
                **terrain_gate_params,
                "command_name": "base_velocity",
                "threshold": 0.35,
                "sensor_cfg": _FEET_SENSOR_CFG,
            },
        )
        self.rewards.rough_alternating_height = RewTerm(
            func=mdp.terrain_gated_alternating_foot_height,
            weight=3.0,
            params={
                **terrain_gate_params,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_leg_4"]),
                "cycle_time": 0.8,
                "target_clearance": 0.08,
                "clearance_margin": 0.03,
                "max_clearance": 0.22,
                "std": 0.05,
            },
        )
        self.rewards.rough_swing_contact = RewTerm(
            func=mdp.terrain_gated_swing_contact,
            weight=-1.0,
            params={
                **terrain_gate_params,
                "sensor_cfg": _FEET_SENSOR_CFG,
                "cycle_time": 0.8,
                "sigma": 25.0,
            },
        )
        self.rewards.both_feet_airborne = RewTerm(
            func=mdp.both_feet_airborne,
            weight=-2.0,
            params={"sensor_cfg": _FEET_SENSOR_CFG},
        )


@configclass
class MiniStairsJumpEnvCfg_PLAY(MiniStairsJumpEnvCfg):
    """小规模、无域随机化的轮腿混合楼梯回放环境。"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 25
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 20
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = True

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None

        # Without --keyboard, use a fixed command so the wheel-to-gait
        # transition can be observed.
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class MiniStairsJumpLegacyEnvCfg_PLAY(MiniStairsJumpEnvCfg_PLAY):
    """Compatibility playback for the legacy 36/235-D perception-jump policy."""

    def __post_init__(self):
        super().__post_init__()

        # The 2026-08-03 policy predates the deployable two-dimensional gait
        # phase observation.  Removing it restores the 36-D Actor input and
        # the original observation ordering.
        self.observations.policy.gait_phase = None
        self.observations.critic.gait_phase = None

        # Restore the scanner-driven hidden FSM used by the legacy training
        # run.  It never entered the Actor observation, but its four-state
        # summary was the final term of the Critic observation.
        self.commands.terrain_jump = mdp.TerrainJumpCommandCfg(
            sensor_cfg=SceneEntityCfg("contact_forces", body_names=[".*_leg_4"]),
            terrain_sensor_cfg=SceneEntityCfg("height_scanner"),
            s1_timeout_s=1.2,
            s3_timeout_s=0.8,
            recovery_resample_s=1.0,
            flight_reward_window_s=0.5,
            target_height=0.65,
            min_obstacle_height=0.025,
            max_obstacle_height=0.35,
            min_detection_distance=0.2,
            max_detection_distance=0.7,
            takeoff_distance=0.35,
            prep_timeout_s=2.0,
        )
        self.observations.critic.jump_state = ObsTerm(
            func=mdp.jump_state,
            params={"command_name": "terrain_jump"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
