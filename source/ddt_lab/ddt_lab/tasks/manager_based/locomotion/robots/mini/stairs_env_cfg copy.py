# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mini 机器人的低难度爬楼梯训练任务。

本任务继承 ``MiniRecoveryRoughEnvCfg`` 的摔倒恢复奖励和 Critic 高度扫描器，
并将默认的混合崎岖地形替换为单一的倒金字塔台阶地形。机器人从中央低平台
出发，沿世界坐标系 +X 方向向外运动时逐级爬升。
"""

from __future__ import annotations
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

from .recovery_env_cfg import MiniRecoveryRoughEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def stairs_terrain_levels(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    success_distance: float = 1.6,
    failure_distance: float = 1.2,
) -> torch.Tensor:
    """根据机器人向前爬行的距离升降地形等级。

    父类崎岖地形课程使用半个地形长度（8 m 地形对应 4 m）作为升级阈值，
    对入门楼梯任务过难。因此这里把到达前几级台阶视为成功：

    - 向 +X 前进超过 ``success_distance``：地形等级 +1；
    - 向 +X 前进不足 ``failure_distance``：地形等级 -1；
    - 其他情况：维持当前等级。
    """

    # 机器人世界坐标减去当前地形原点，得到相对地形原点的 +X 前进距离。
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    forward_distance = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]

    # 同一次更新中升级优先，避免两个条件意外重叠时同时升降级。
    move_up = forward_distance > success_distance
    move_down = forward_distance < failure_distance
    move_down &= ~move_up

    # Isaac Lab 会将等级限制在合法范围，并更新这些环境的地形出生点。
    terrain.update_env_origins(env_ids, move_up, move_down)
    # 返回平均等级，供课程管理器记录到训练日志。
    return torch.mean(terrain.terrain_levels.float())


@configclass
class MiniStairsEnvCfg(MiniRecoveryRoughEnvCfg):
    """继承 Mini 崎岖地形恢复任务的爬楼梯训练环境。"""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------ #
        # 地形：只保留倒金字塔台阶。其地形原点位于中央最低平台，机器人从       #
        # 原点向外走就是上楼；普通 pyramid_stairs 则会从中央高台向外下楼。     #
        # ------------------------------------------------------------------ #
        terrain_cfg = self.scene.terrain.terrain_generator
        if terrain_cfg is None:
            raise RuntimeError("MiniStairsEnvCfg requires generated terrain.")

        stairs_cfg = terrain_cfg.sub_terrains["pyramid_stairs_inv"]
        stairs_cfg.proportion = 1.0                 # 台阶地形占比 100%
        stairs_cfg.step_height_range = (0.03, 0.3)  # 台阶高度随难度从 3 cm 增至 30 cm
        stairs_cfg.step_width = 0.40                # 每级踏步在水平方向宽 40 cm
        stairs_cfg.platform_width = 2.0             # 中央低平台宽 2 m，作为出生区域
        stairs_cfg.border_width = 1.0               # 地形外围保留 1 m 边界
        stairs_cfg.holes = False                    # 生成实体台阶，不在台阶之间留空洞

        terrain_cfg.sub_terrains = {"stairs_up": stairs_cfg}  # 删除其余混合崎岖地形
        terrain_cfg.num_rows = 20                   # 20 行分别对应 20 个难度等级
        terrain_cfg.num_cols = 10                   # 每个等级生成 10 个随机地形样本
        terrain_cfg.difficulty_range = (0.0, 1.0)   # 使用完整的 0%～100% 难度范围
        terrain_cfg.curriculum = True               # 按行从易到难生成课程地形

        # ------------------------------------------------------------------ #
        # 课程：训练时所有机器人只从第 0 级开始。第 0 级接近 3 cm，随着前进     #
        # 成功逐级提升，最高等级的台阶高度接近 20 cm。                         #
        # ------------------------------------------------------------------ #
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = CurrTerm(
            func=stairs_terrain_levels,
            params={
                "success_distance": 1.6,  # 向 +X 前进超过 1.6 m 时升级
                "failure_distance": 1.2,  # 向 +X 前进不足 1.2 m 时降级
            },
        )

        # ------------------------------------------------------------------ #
        # 速度指令：只要求沿机器人前方行走，并通过 heading 控制使机器人基本     #
        # 朝向世界坐标 +X。ang_vel_z 是 heading 控制器输出的角速度上下限。      #
        # ------------------------------------------------------------------ #
        command_cfg = self.commands.base_velocity
        command_cfg.rel_standing_envs = 0.0          # 不生成原地站立指令
        command_cfg.rel_heading_envs = 1.0           # 所有环境都使用目标朝向控制
        command_cfg.heading_command = True           # 启用绝对 heading 目标
        command_cfg.ranges.lin_vel_x = (0.2, 0.60)  # 前进速度 0.2～0.6 m/s
        command_cfg.ranges.lin_vel_y = (0.0, 0.0)    # 禁止横向速度
        command_cfg.ranges.ang_vel_z = (-.4, 0.4)  # 转向角速度限制 ±0.4 rad/s
        command_cfg.ranges.heading = (-0.1, 0.1)   # 朝向约束在 +X 附近 ±0.1 rad

        # ------------------------------------------------------------------ #
        # 重置：相比完整摔倒恢复任务更容易。机器人从中央低平台附近、接近直立     #
        # 且速度较小的状态开始，避免训练初期同时学习起身和爬楼。                #
        # pose_range 的 xyz 是相对机器人默认根位置的偏移。                    #
        # ------------------------------------------------------------------ #
        self.events.reset_base.params = {
            # "pose_range": {
            #     "x": (-0.15, 0.15),       # 出生点前后随机偏移 ±15 cm
            #     "y": (-0.15, 0.15),       # 出生点左右随机偏移 ±15 cm
            #     "z": (0.0, 0.05),         # 在默认根高度上增加 0～5 cm
            #     "roll": (-0.10, 0.10),    # 横滚扰动约 ±5.7°
            #     "pitch": (-0.10, 0.10),   # 俯仰扰动约 ±5.7°
            #     "yaw": (-0.10, 0.10),     # 偏航扰动约 ±5.7°
            # },
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-0.0, 0.0),
                "pitch": (-0, 0),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),       # X 方向初速度
                "y": (-0.05, 0.05),       # Y 方向初速度
                "z": (-0.05, 0.05),       # Z 方向初速度
                "roll": (-0.10, 0.10),    # 横滚角速度
                "pitch": (-0.10, 0.10),   # 俯仰角速度
                "yaw": (-0.10, 0.10),     # 偏航角速度
            },
        }

        # ------------------------------------------------------------------ #
        # 鲁棒性：采用 D1 Rough 的 reset 外力和周期速度推搡。                   #
        # 越界条件与 D1 一致，检测的是整张生成地形地图的边缘，而不是单个        #
        # 8 m × 8 m 楼梯子地形的局部边缘。                                    #
        # ------------------------------------------------------------------ #
        self.terminations.terrain_out_of_bounds = DoneTerm(
            func=mdp.terrain_out_of_bounds,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "distance_buffer": 3.0,
            },
            time_out=True,
        )
        self.events.base_external_force_torque = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "force_range": (-3.0, 3.0),
                "torque_range": (-3.0, 3.0),
            },
        )
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "velocity_range": {
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                    "z": (-0.1, 0.1),
                }
            },
        )

        # ------------------------------------------------------------------ #
        # 奖励：提高向前速度跟踪权重；基座高度相对扫描到的台阶表面计算，避免     #
        # 把机器人爬升后的世界坐标高度误认为高度偏差。                          #
        # 关闭左右关节镜像惩罚，使两条腿可以采用非对称动作跨越台阶。             #
        # ------------------------------------------------------------------ #
        self.rewards.track_lin_vel_xy_exp.weight = 2  # 奖励跟踪前进速度
        self.rewards.track_ang_vel_z_exp.weight = 1   # 奖励保持目标朝向
        self.rewards.base_height_l2.weight = -2.0       # 惩罚相对地面的基座高度误差
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.rewards.joint_mirror.weight = -0.2         # 允许左右腿执行不同动作


@configclass
class MiniStairsEnvCfg_PLAY(MiniStairsEnvCfg):
    """用于可视化评估的较小、无扰动场景。"""

    def __post_init__(self):
        super().__post_init__()

        # 评估时使用 25 个并行环境；None 表示初始等级可覆盖全部 20 级。
        self.scene.num_envs = 25
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 20  # 保留完整的 20 级难度
            self.scene.terrain.terrain_generator.num_cols = 5   # 减少列数以节省显存
            self.scene.terrain.terrain_generator.curriculum = True

        # 关闭观测噪声和域随机化，便于稳定复现并观察策略本身的表现。
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
