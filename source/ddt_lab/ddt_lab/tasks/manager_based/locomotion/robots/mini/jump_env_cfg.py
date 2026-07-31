# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mini 平地“摔倒恢复 + 指令跳跃”联合训练环境。

``jump_cmd`` 在每个采样段中取 0/1：当前配置下
``P(jump_cmd=1)=0.2``、``P(jump_cmd=0)=0.8``。不跳样本用于保留自起身和
站稳能力；跳跃样本依次经过以下状态机：

* PREP（蓄力）：``jump_before_setting``，当前权重 +2.0。
* TAKEOFF/FLIGHT（起跳/飞行）：``lin_vel_z_jump``，当前权重 +18.0。
* FLIGHT 短窗口（高度）：``jump_flight_height``，当前权重 +28.0。
* LAND（落地）：``jump_land_stability`` +5.0，
  ``jump_land_orientation`` +3.0。

跳跃指令由 ``mdp.JumpCommandCfg`` 管理：第 125 个控制步触发，
TAKEOFF 最长等待 5 s，LAND 恢复段持续 3 s 后重新采样。训练中不施加
额外 z 向速度或外力，跳跃必须由策略自己学会。

详细调参方法见同目录的 ``jump_env_cfg_tuning_zh.md``。

直接训练：
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
    """Mini 平地跳跃训练环境。"""

    def __post_init__(self):
        super().__post_init__()

        # 单回合最长 15 s。父类的控制周期是 0.02 s，即最多 750 个控制步。
        self.episode_length_s = 15.0
        # 允许 base_link 接触地面后继续学习自起身，不因触地立即终止回合。
        self.terminations.base_contact = None

        # 跳跃指令状态机。每个参数的调整顺序和建议范围见调参文档。
        self.commands.jump_cmd = mdp.JumpCommandCfg(
            # torch.randint 的上界不包含，(125, 126) 等价于固定在第 125 步触发。
            # 控制周期 0.02 s，因此当前蓄力时间约为 2.5 s。
            trigger_step_range=(125, 126),
            # 前 500 个训练迭代只学恢复/站稳，不进入跳跃阶段。
            warmup_iterations=500,
            # 必须与 NP3O runner.num_steps_per_env 保持一致，否则 warmup 换算会失真。
            steps_per_iteration=24,
            # 目标 COM 高度；用于高度奖励归一化和 critic 状态，不是强制终止线。
            target_height=0.8,
            # 每次重采样时有 20% 的概率产生 jump_cmd=1。
            jump_probability=0.2,
            # jump_cmd=0 的 RECOVERY 段持续 3 s 后重新抽样。
            recovery_resample_s=3.0,
            # 只在进入 FLIGHT 后的前 0.8 s 发放 jump_flight_height。
            flight_reward_window_s=0.8,
            # 用两个 .*_leg_4 轮/足接触判定离地和落地。
            sensor_cfg=SceneEntityCfg("contact_forces", body_names=[".*_leg_4"]),
        )
        # 保留与速度任务/键盘控制一致的 [vx, vy, wz] 命令语义。
        self.commands.base_velocity.heading_command = True
        # 0.0 表示不采样 heading 模式环境，角速度直接由 ang_vel_z 范围采样。
        self.commands.base_velocity.rel_heading_envs = 0.0
        # 跳跃训练时缩小速度范围，避免位移跟踪压过跳跃主任务。
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        # 当前继承父类的 vy 范围；若要禁止横移，显式设为 (0.0, 0.0)。
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # actor 额外观测 1 维 jump_cmd，使策略能区分“站稳/恢复”和“跳跃”指令。
        self.observations.policy.jump_cmd = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "jump_cmd"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # critic 同样观测 jump_cmd。
        self.observations.critic.jump_cmd = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "jump_cmd"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # critic 额外获得 4 维特权状态：归一化 stage、是否离地过、是否已落地、最大高度。
        self.observations.critic.jump_state = ObsTerm(
            func=mdp.jump_state,
            params={"command_name": "jump_cmd"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # ------------------------------------------------------------------ #
        # 全流程初始化：先学习任意姿态恢复，再按指令跳跃                    #
        # ------------------------------------------------------------------ #
        # reset 时位置和姿态覆盖站立、侧躺、倒地和低空状态。范围过大会让
        # 恢复任务占比过高，范围过小则可能丢失摔倒后自起身的能力。
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

        # 禁用任何起跳辅助：不直接改 z 速度，也不施加向上外力。
        self.events.jump_assist = None

        # ------------------------------------------------------------------ #
        # 恢复/站稳奖励：整个回合持续生效                                 #
        # ------------------------------------------------------------------ #

        # base_link 接地始终扣分，但不终止回合，为自起身保留学习时间。
        self.rewards.base_contact_raw.weight = -2.0
        # 关闭父类中带姿态过滤的重复 base 接触罚项，避免双重计算。
        self.rewards.base_contact_penalty.weight = 0.0
        # 线性奖励 -g_z：倒立时仍有恢复梯度，站直时取最大值。
        self.rewards.upright_progress.weight = 2.0
        # 只在倒置时小幅奖励角速度，用于脱离完全倒置死区；过大会学成翻滚刷分。
        self.rewards.inverted_ang_vel_bonus.weight = 0.1
        # 站立时约束 base 高度到父类设定的 0.35 m；该项带直立姿态门控。
        self.rewards.base_height_l2.weight = -10.0

        # 关闭 z 速度惩罚，否则会直接抵消起跳的向上速度奖励。
        self.rewards.lin_vel_z_l2.weight = 0.0

        # 关闭 XY 角速度惩罚，允许腾空期间必要的姿态调整。
        self.rewards.ang_vel_xy_l2.weight = 0.0

        # 保留对 [vx, vy, wz] 的跟踪能力，但降低权重，避免位移任务压过跳跃任务。
        self.rewards.track_lin_vel_xy_exp.weight = 0.6
        self.rewards.track_ang_vel_z_exp.weight = 0.3

        # 保留一定的 roll/pitch 惩罚：允许飞行中倾斜，但抑制地面极端姿态。
        self.rewards.flat_orientation_l2.weight = -5.0

        # ------------------------------------------------------------------ #
        # 通用直立奖励：让 jump_cmd=0 时保持站稳                         #
        # ------------------------------------------------------------------ #
        # upward=(1-g_z)^2：站直约为 4，侧倒约为 1，完全倒置为 0。
        self.rewards.upward.weight = 1.0

        # ------------------------------------------------------------------ #
        # PREP：起跳前下蹲蓄力                                           #
        # ------------------------------------------------------------------ #
        # 以 crouch_height=0.25 m 为中心的高斯奖励（正常站立高度约 0.35 m），
        # 仅在 PREP 且至少一只轮/足接地时生效。
        self.rewards.jump_before_setting.weight = 2.0
        self.rewards.jump_before_setting.params["command_name"] = "jump_cmd"

        # ------------------------------------------------------------------ #
        # TAKEOFF/FLIGHT：奖励世界系向上 z 速度                            #
        # ------------------------------------------------------------------ #
        # 通过 jump_cmd 状态机门控后，奖励在 TAKEOFF 和 FLIGHT 两个阶段生效，
        # 从脚还在地面时就提供起跳梯度，避免只有偶然离地才能得分。
        self.rewards.lin_vel_z_jump.weight = 18.0
        # 当 command_name 有效时，实际门控由 FSM stage 完成；
        # 此值保持 False 也与无 FSM 回退语义一致。
        self.rewards.lin_vel_z_jump.params["require_in_air"] = False
        self.rewards.lin_vel_z_jump.params["command_name"] = "jump_cmd"

        # ------------------------------------------------------------------ #
        # FLIGHT：在离地后的短窗口内奖励 COM 高度                         #
        # ------------------------------------------------------------------ #
        # 当前是主要跳高信号，仅在 flight_reward_window_s 指定的早期飞行窗口生效。
        self.rewards.jump_flight_height.weight = 28.0
        self.rewards.jump_flight_height.params["command_name"] = "jump_cmd"
        # 目标高度与 JumpCommandCfg.target_height 保持一致。
        self.rewards.jump_flight_height.params["target_height"] = 0.8
        # COM 低于 0.42 m 时不发放高度奖励，防止轻微伸腿被当成跳跃。
        self.rewards.jump_flight_height.params["min_height"] = 0.42

        # ------------------------------------------------------------------ #
        # LAND：奖励落地后低滚转速度和直立姿态                         #
        # ------------------------------------------------------------------ #
        # 返回 exp(-roll_rate^2-pitch_rate^2)，因此这里应使用正权重；越稳定越接近 1。
        self.rewards.jump_land_stability.weight = 5.0
        self.rewards.jump_land_stability.params["command_name"] = "jump_cmd"

        # 返回 roll/pitch 倾斜误差的指数奖励，仅在 LAND 且接地时生效。
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
    """Mini 跳跃策略回放配置：固定初始姿态，关闭训练随机化。"""

    def __post_init__(self):
        super().__post_init__()

        # 回放只需少量并行环境，增大间距便于观察。
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 回放时关闭 actor 观测噪声，保证行为可重复。
        self.observations.policy.enable_corruption = False

        # 关闭外力、推力、质量/COM/惯量和执行器增益随机化，用于检查策略本身。
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.jump_assist = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None

        # play 模式下默认关闭随机速度命令；使用 --keyboard 时再由键盘写入 [vx, vy, wz]。
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1e9, 1e9)

        # play_mode 会把自动采样的 jump_cmd 固定为 1，250 步约等于 5 s 后触发。
        # 若启用 --keyboard，play.py 会切换为 manual_mode，此时由 R 键控制跳跃指令。
        self.commands.jump_cmd.play_mode = True
        self.commands.jump_cmd.play_trigger_steps = 250
        # 回放不需要训练 warmup。
        self.commands.jump_cmd.warmup_iterations = 0

        # 固定为地面直立静止初始状态，便于可重复地评估起跳和落地。
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
