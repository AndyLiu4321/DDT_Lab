# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mini fall-recovery env (flat terrain, NP3O constrained).

Design philosophy — adapted from gym_np3o/legged_robot.py:

  base 触地 → 扣分但不重置
  ────────────────────────────────────────────────────────────
  * base_contact termination is REMOVED so the robot experiences the
    fallen state and must learn to self-right.
  * base_contact_raw (-2.0/step): base contact penalty WITHOUT the
    uprightness filter, so it fires even when fully inverted.
  * upward (+1.0): quadratic (1-g_z)², large gradient when nearly upright.
  * upright_progress (+2.0): linear -g_z, CONSTANT gradient everywhere
    including at g_z=+1 (fully inverted dead zone where upward → 0).
  * inverted_ang_vel_bonus (+0.5): rewards angular velocity scaled by how
    inverted the robot is → encourages aggressive rolling to escape inversion.
  * flat_orientation_l2 (-5.0) and base_height_l2 (-10.0) maintain
    the orientation and height signal throughout the episode.
  * Velocity-tracking rewards are suppressed (weight=0) — impossible
    to follow speed commands while fallen, and the conflicting gradient
    slows recovery.

Reset: roll ±180°, pitch ±90° — covers all orientations including
fully inverted (g_z=+1), which the original ±90°/±90° could miss.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass  # noqa: F401 — resolved at runtime via Isaac Lab install

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from .flat_env_cfg import MiniFlatEnvCfg
from .rough_env_cfg import MiniRoughEnvCfg

_90_DEG = 1.5708  # radians


@configclass
class MiniRecoveryEnvCfg(MiniFlatEnvCfg):
    """Mini flat-terrain fall-recovery environment."""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------ #
        # Termination: NO reset on base contact                               #
        # Robot stays in fallen state and must learn to self-right.           #
        # ------------------------------------------------------------------ #
        self.terminations.base_contact = None

        # ------------------------------------------------------------------ #
        # Reset: ±90° roll / pitch — robot starts on its side ~half the time  #
        # ------------------------------------------------------------------ #
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.05, 0.5),
                "roll":  (-3.14, 3.14),   # ±180°: covers fully inverted (was ±90°)
                "pitch": (-_90_DEG, _90_DEG),
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

        # ------------------------------------------------------------------ #
        # 奖励函数配置                                                        #
        # ------------------------------------------------------------------ #

        # ── 恢复核心信号（始终有效）─────────────────────────────────────────

        # 【原始接触惩罚】无过滤版：完全翻倒时 undesired_contacts 的
        # clamp(-g_z,0,0.7)/0.7 过滤器会把惩罚归零，这里用无过滤版
        # 确保翻倒时 base 接触地面依然产生梯度
        self.rewards.base_contact_raw.weight = -2.0

        # 【线性恢复梯度】-g_z ∈ [-1, +1]：在任意姿态都有常数梯度 -1
        # 解决 upward=(1-g_z)² 在 g_z=+1 时梯度归零的死区问题
        # 完全翻倒时给出 -1 惩罚，驱动机器人向减小 g_z 的方向运动
        self.rewards.upright_progress.weight = 2.0

        # 【二次站立奖励】(1 - g_z)²：站直时 4，侧躺时 1，倒扣时 0
        # 与 upright_progress 组合，接近站立时给出更强的二次梯度
        self.rewards.upward.weight = 1.0

        # 【翻倒角速度奖励】翻倒越深奖励越大的角速度，鼓励做激进翻滚动作
        # inverted_weight = clamp(g_z, 0, 1)，完全翻倒时权重最大
        self.rewards.inverted_ang_vel_bonus.weight = 0.5

        # 【姿态惩罚】惩罚重力投影在 xy 方向的偏移，迫使机身保持水平
        # 注：此项有过滤器，翻倒时会减弱，由 upright_progress 补偿
        self.rewards.flat_orientation_l2.weight = -5.0

        # 【高度惩罚】惩罚质心高度偏离目标 0.35 m，防止机器人方向对了但贴地趴
        self.rewards.base_height_l2.weight = -10.0

        # 【原始接触惩罚（旧版）】有过滤器版保持关闭，改用 base_contact_raw
        self.rewards.base_contact_penalty.weight = 0.0

        # ── 屏蔽与恢复无关的速度指令信号 ──────────────────────────────────

        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # ── CartPole 风格存活信号（无实际终止条件时无效，保持关闭）────────

        self.rewards.alive.weight = 0.0
        self.rewards.is_terminated.weight = 0.0


@configclass
class MiniRecoveryRoughEnvCfg(MiniRoughEnvCfg):
    """Mini 崎岖地形摔倒恢复环境。

    与 Flat 版本的唯一区别：保留粗糙地形（摩擦/高度随机性），
    但禁用高度扫描传感器，使观测维度与 Flat 版完全一致（critic=54），
    可以直接加载 Flat Recovery 的 checkpoint 继续训练。
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------ #
        # 关闭 scanner：与 MiniFlatEnvCfg 对齐                               #
        # 原因：                                                               #
        #   1. 禁用后 critic 维度从 241 降回 54，与 Flat 版一致               #
        #   2. 不再需要 scan_encoder，可直接加载 Flat checkpoint              #
        #   3. 恢复训练阶段不需要高度感知，减少无效观测噪声                     #
        # ------------------------------------------------------------------ #
        self.scene.height_scanner = None          # 关闭传感器
        self.observations.scanner = None          # 关闭 scanner 观测组

        # ------------------------------------------------------------------ #
        # 地形难度课程：从最低级 (0) 开始，distance-based 升降级               #
        # 走超过格子长度一半(4m)→升级，走不够指令速度期望距离一半→降级          #
        # terrain_generator.curriculum=True 已由 super().__post_init__() 设置  #
        # ------------------------------------------------------------------ #
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_recovery)
        self.scene.terrain.max_init_terrain_level = 0  # 只在第 0 级（最平坦）初始化

        # ------------------------------------------------------------------ #
        # 终止条件：base 触地不重置，允许机器人在地上挣扎并学习站起来           #
        # ------------------------------------------------------------------ #
        self.terminations.base_contact = None

        # ------------------------------------------------------------------ #
        # 重置：±90° roll/pitch，约一半 episode 从侧躺状态开始                #
        # z 范围覆盖站立（≈0.35m）和倒地（≈0.1m）两种初始高度               #
        # ------------------------------------------------------------------ #
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.05, 0.5),
                "roll":  (-3.14, 3.14),   # ±180°: covers fully inverted
                "pitch": (-_90_DEG, _90_DEG),
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

        # ------------------------------------------------------------------ #
        # 奖励函数配置（与平地恢复版一致，含翻倒死区修复）                     #
        # ------------------------------------------------------------------ #

        # 【原始接触惩罚】无过滤版，翻倒时依然产生梯度
        self.rewards.base_contact_raw.weight = -2.0
        self.rewards.base_contact_penalty.weight = 0.0  # 关闭有过滤器的旧版

        # 【线性恢复梯度】任意姿态恒定梯度，解决完全翻倒时梯度归零问题
        self.rewards.upright_progress.weight = 2.0

        # 【二次站立奖励】接近直立时提供更强梯度
        self.rewards.upward.weight = 1.0

        # 【翻倒角速度奖励】翻倒越深越鼓励大幅翻滚动作
        self.rewards.inverted_ang_vel_bonus.weight = 0.5

        # 【姿态惩罚】有过滤器，翻倒时减弱，由 upright_progress 补偿
        self.rewards.flat_orientation_l2.weight = -1.0

        # 【高度惩罚】防止方向对了但贴地趴
        self.rewards.base_height_l2.weight = -7.0

        # 【速度指令】倒地时无法跟踪，关闭避免梯度冲突
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # CartPole 风格信号（无有效终止条件，保持关闭）
        self.rewards.alive.weight = 0.0
        self.rewards.is_terminated.weight = 0.0


@configclass
class MiniRecoveryRoughEnvCfg_PLAY(MiniRecoveryRoughEnvCfg):
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


@configclass
class MiniRecoveryEnvCfg_PLAY(MiniRecoveryEnvCfg):
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
