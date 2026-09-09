# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Andy rough-terrain locomotion configuration.

Andy has six actuated joints and two passive support-wheel joints.  The MDP is
based on Mini's six-action task, while all joint/body selectors are adjusted to
match ``andymini.urdf``.
"""

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ddt_lab.assets.ddt_robot import DDT_ANDY_CFG
from ddt_lab.tasks.manager_based.locomotion.robots.mini.rough_env_cfg import (
    ActionsCfg as MiniActionsCfg,
    MiniRoughEnvCfg,
    SceneCfg as MiniSceneCfg,
)


ANDY_ACTUATED_JOINTS = [
    "joint_left_leg_1",
    "joint_left_leg_2",
    "joint_left_leg_3",
    "joint_right_leg_1",
    "joint_right_leg_2",
    "joint_right_leg_3",
]
ANDY_POSITION_JOINTS = [
    "joint_left_leg_1",
    "joint_left_leg_2",
    "joint_right_leg_1",
    "joint_right_leg_2",
]
ANDY_DRIVE_JOINTS = ["joint_left_leg_3", "joint_right_leg_3"]
ANDY_DRIVE_BODIES = ["left_leg_3", "right_leg_3"]
ANDY_AUXILIARY_WHEEL_BODIES = ["left_wheel_1", "right_wheel_1"]
ANDY_NON_DRIVE_LEG_BODIES = [".*_leg_(1|2)"]


@configclass
class AndySceneCfg(MiniSceneCfg):
    """Mini scene with the Andy articulation."""

    robot: ArticulationCfg = DDT_ANDY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class AndyActionsCfg(MiniActionsCfg):
    """Six motor commands in hardware order: left 1/2/3, right 1/2/3."""

    joint_pos_0 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_left_leg_1"],
        scale=0.35,
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )
    joint_pos_1 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_left_leg_2"],
        scale=0.35,
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )
    joint_vel_2 = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["joint_left_leg_3"],
        scale=12.0,
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )
    joint_pos_3 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_right_leg_1"],
        scale=0.35,
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )
    joint_pos_4 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_right_leg_2"],
        scale=0.35,
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )
    joint_vel_5 = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["joint_right_leg_3"],
        scale=12.0,
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )


def configure_andy_mdp(env_cfg) -> None:
    """Retarget inherited Mini MDP selectors to Andy's URDF names."""

    # SceneEntityCfg is resolved in-place by each manager.  Always create a
    # fresh selector per term instead of sharing one resolved instance.
    def actuated_cfg() -> SceneEntityCfg:
        return SceneEntityCfg("robot", joint_names=ANDY_ACTUATED_JOINTS, preserve_order=True)

    def drive_joint_cfg() -> SceneEntityCfg:
        return SceneEntityCfg("robot", joint_names=ANDY_DRIVE_JOINTS)

    # Policy and critic deliberately expose only the six motor joints.  This
    # preserves Mini's 27-D actor layout and excludes the two passive joints.
    for group in (env_cfg.observations.policy, env_cfg.observations.critic):
        group.joint_pos.params["asset_cfg"] = actuated_cfg()
        group.joint_pos.params["wheel_asset_cfg"] = drive_joint_cfg()
        group.joint_vel.params["asset_cfg"] = actuated_cfg()

    env_cfg.observations.priv.contact_state.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=ANDY_DRIVE_BODIES
    )
    env_cfg.observations.priv.joint_kp_factor.params["asset_cfg"] = actuated_cfg()
    env_cfg.observations.priv.joint_kd_factor.params["asset_cfg"] = actuated_cfg()

    env_cfg.events.randomize_actuator_gains.params["asset_cfg"] = actuated_cfg()
    env_cfg.events.reset_robot_joints.params["asset_cfg"] = actuated_cfg()

    env_cfg.rewards.dof_torques_l2.params["asset_cfg"] = actuated_cfg()
    env_cfg.rewards.dof_acc_l2.params["asset_cfg"] = actuated_cfg()
    env_cfg.rewards.joint_mirror.params["mirror_joints"] = [
        ["joint_left_leg_(1|2)", "joint_right_leg_(1|2)"]
    ]
    env_cfg.rewards.stand_still.params["asset_cfg"] = drive_joint_cfg()
    env_cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=ANDY_NON_DRIVE_LEG_BODIES
    )
    env_cfg.rewards.base_height_l2.params["target_height"] = 0.32

    # Zero-weight jump terms are still resolved by RewardManager, so their
    # contact selectors also have to reference existing Andy bodies.
    for name in (
        "jump_before_setting",
        "lin_vel_z_jump",
        "jump_flight_height",
        "jump_land_stability",
        "jump_land_orientation",
    ):
        term = getattr(env_cfg.rewards, name, None)
        if term is not None and "sensor_cfg" in term.params:
            term.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=ANDY_DRIVE_BODIES)

    env_cfg.costs.joint_pos_limit.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=ANDY_POSITION_JOINTS
    )
    env_cfg.costs.joint_vel_limit.params["asset_cfg"] = actuated_cfg()
    env_cfg.costs.joint_torque_limit.params["asset_cfg"] = actuated_cfg()


@configclass
class AndyRoughEnvCfg(MiniRoughEnvCfg):
    scene: AndySceneCfg = AndySceneCfg(num_envs=4096, env_spacing=2.5)
    actions: AndyActionsCfg = AndyActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_andy_mdp(self)

        # The inherited Mini event writes a random +/-10 N / +/-10 N*m wrench
        # at reset.  External wrenches remain active on the articulation until
        # overwritten, so this becomes a continuous disturbance and launches
        # the much lighter Andy robot after wheel/ground impacts.  The Andy
        # reference configuration has push_robots=False.
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AndyRoughEnvCfg_PLAY(AndyRoughEnvCfg):
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
