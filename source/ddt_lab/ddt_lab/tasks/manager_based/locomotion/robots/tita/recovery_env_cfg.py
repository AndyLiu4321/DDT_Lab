# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tita fall-recovery env (flat terrain, NP3O constrained).

Design philosophy — adapted from gym_np3o/legged_robot.py:

  base 触地 → 扣分但不重置
  ────────────────────────────────────────────────────────────
  * base_contact termination is REMOVED so the robot experiences the
    fallen state and must learn to self-right.
  * base_contact_penalty (-2.0/step) fires every step the base is on
    the ground — analogous to legged_robot._reward_termination but
    continuous rather than one-shot.
  * upward (+1.0) provides a dense linear-style recovery gradient:
    larger when more upright.
  * flat_orientation_l2 (-5.0) and base_height_l2 (-10.0) maintain
    the orientation and height signal throughout the episode.
  * Velocity-tracking rewards are suppressed (weight=0) — impossible
    to follow speed commands while fallen, and the conflicting gradient
    slows recovery.

Reset: full ±90° roll/pitch so roughly half the episodes start on
the robot's side, requiring active recovery.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass  # noqa: F401 — resolved at runtime via Isaac Lab install

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from .flat_env_cfg import TitaFlatEnvCfg
from .rough_env_cfg import TitaRoughEnvCfg

_90_DEG = 1.5708  # radians


@configclass
class TitaRecoveryEnvCfg(TitaFlatEnvCfg):
    """Tita flat-terrain fall-recovery environment."""

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
                "roll":  (-_90_DEG, _90_DEG),
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
        # Command: keep the same base_velocity interface as locomotion/play,  #
        # but start with a conservative forward-speed range like Mini jump.   #
        # ------------------------------------------------------------------ #

        # ------------------------------------------------------------------ #
        # 奖励函数配置                                                        #
        # ------------------------------------------------------------------ #

        # ── 恢复核心信号（始终有效）─────────────────────────────────────────

        # 【接触惩罚】base_link 每帧接触地面扣 2 分，但不重置 episode
        # 参考 gym_np3o/legged_robot._reward_termination（连续版本）
        # 函数: mdp.undesired_contacts
        #   → ddt_lab/tasks/manager_based/locomotion/mdp/rewards.py:665
        self.rewards.base_contact_penalty.weight = -2.0

        # 【站立奖励】(1 - g_z)^2：站直时 4，侧躺时 1，倒扣时 0
        # 提供密集的"站起来"梯度信号
        # 函数: mdp.upward
        #   → ddt_lab/tasks/manager_based/locomotion/mdp/rewards.py:608
        self.rewards.upward.weight = 1.0

        # 【姿态惩罚】惩罚重力投影在 xy 方向的偏移，迫使机身保持水平
        # = -5.0 * sum(projected_gravity_b[:2]^2)，站直时接近 0
        # 函数: mdp.flat_orientation_l2
        #   → ddt_lab/tasks/manager_based/locomotion/mdp/rewards.py:678
        self.rewards.flat_orientation_l2.weight = -5.0

        # 【高度惩罚】惩罚质心高度偏离目标 0.35 m，防止机器人方向对了但贴地趴
        # = -10.0 * (height - 0.35)^2，比正常训练的 -2.0 强 5 倍
        # 函数: mdp.base_height_l2
        #   → ddt_lab/tasks/manager_based/locomotion/mdp/rewards.py:616
        self.rewards.base_height_l2.weight = -10.0

        # ── 屏蔽与恢复无关的速度指令信号 ──────────────────────────────────

        # 【线速度跟踪】倒地时无法执行速度指令，保留会产生矛盾梯度 → 关闭
        # 函数: mdp.track_lin_vel_xy_exp
        #   → ddt_lab/tasks/manager_based/locomotion/mdp/rewards.py:24
        self.rewards.track_lin_vel_xy_exp.weight = 0.0

        # 【角速度跟踪】同上 → 关闭

        # 【角速度跟踪】先关闭，只训练站起后的线速度；稳定后再逐步打开
        # 函数: mdp.track_ang_vel_z_exp
        #   → ddt_lab/tasks/manager_based/locomotion/mdp/rewards.py:40
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # ── CartPole 风格存活信号（无实际终止条件时无效，保持关闭）────────

        # 函数: isaaclab.envs.mdp.is_alive / is_terminated
        #   → IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:31,36
        self.rewards.alive.weight = 0.0
        self.rewards.is_terminated.weight = 0.0


@configclass
class TitaRecoveryRoughEnvCfg(TitaRoughEnvCfg):
    """Tita 崎岖地形摔倒恢复环境。

    与 Flat 版本的唯一区别：保留粗糙地形（摩擦/高度随机性），
    但禁用高度扫描传感器，使观测维度与 Flat 版完全一致（critic=54），
    可以直接加载 Flat Recovery 的 checkpoint 继续训练。
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------ #
        # 关闭 scanner：与 TitaFlatEnvCfg 对齐                               #
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
                "roll":  (-_90_DEG, _90_DEG),
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
        # 奖励函数配置（与平地恢复版完全一致）                                 #
        # ------------------------------------------------------------------ #

        # 【接触惩罚】base_link 每帧触地扣 2 分，不重置
        # 函数: mdp.undesired_contacts → mdp/rewards.py:665
        self.rewards.base_contact_penalty.weight = -2.0  # -2.0 → -0.5

        # 【站立奖励】(1 - g_z)^2：站直时 4，侧躺时 1，倒扣时 0
        # 函数: mdp.upward → mdp/rewards.py:608
        self.rewards.upward.weight = 1.0

        # 【姿态惩罚】惩罚重力投影 xy 分量偏移
        # 函数: mdp.flat_orientation_l2 → mdp/rewards.py:678
        self.rewards.flat_orientation_l2.weight = -1.0 # -5.0 → -1.0

        # 【高度惩罚】惩罚质心偏离目标 0.35m，防止方向对了但贴地趴
        # 函数: mdp.base_height_l2 → mdp/rewards.py:616
        self.rewards.base_height_l2.weight = -7.0   # -10.0 → -2.0

        # 【速度指令】倒地时无法跟踪，关闭避免梯度冲突
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # CartPole 风格信号（无有效终止条件，保持关闭）
        self.rewards.alive.weight = 0.0
        self.rewards.is_terminated.weight = 0.0


@configclass
class TitaRecoveryRoughEnvCfg_PLAY(TitaRecoveryRoughEnvCfg):
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
class TitaRecoveryEnvCfg_PLAY(TitaRecoveryEnvCfg):
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
