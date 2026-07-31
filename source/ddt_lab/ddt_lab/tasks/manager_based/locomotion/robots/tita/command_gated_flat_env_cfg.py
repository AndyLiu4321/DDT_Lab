# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .flat_env_cfg import TitaFlatEnvCfg
from .rough_env_cfg import ActionsCfg, CostsCfg, EventCfg, SceneCfg


COMMAND_NAME = "wheel_commands"
LEG_JOINTS = [
    "joint_left_leg_1",
    "joint_left_leg_2",
    "joint_left_leg_3",
    "joint_right_leg_1",
    "joint_right_leg_2",
    "joint_right_leg_3",
]
ALL_JOINTS = [
    "joint_left_leg_1",
    "joint_left_leg_2",
    "joint_left_leg_3",
    "joint_left_leg_4",
    "joint_right_leg_1",
    "joint_right_leg_2",
    "joint_right_leg_3",
    "joint_right_leg_4",
]


@configclass
class CommandGatedCommandsCfg:
    """Six-dimensional command-conditioned single-policy command."""

    wheel_commands = mdp.WheelLeggedCommandCfg(
        asset_cfg=SceneEntityCfg("robot"),
        resampling_time_range=(3.0, 3.0),
        lin_vel_x_range=(-2.0, 2.0),
        lin_vel_y_range=(0.0, 0.0),
        ang_vel_z_range=(-12.0, 12.0),
        leg_length_range=(0.0, 1.0),
        tsk_range=(-0.3, 0.3),
        high_speed=True,
        inverse_linx_angv=1.0,
        inverse_tsk=2.0,
        inverse_leg_length=2.0,
        lin_vel_x_curriculum_ratio=0.3,
        ang_vel_z_curriculum_ratio=0.05,
    )


@configclass
class CommandGatedObservationsCfg:
    """Observation layout: 28-D proprio slice + 6-D command, with policy history."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=0.25
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        wheel_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": COMMAND_NAME},
            clip=(-100.0, 100.0),
            scale=(2.0, 2.0, 0.25, 1.0, 1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        wheel_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": COMMAND_NAME},
            clip=(-100.0, 100.0),
            scale=(2.0, 2.0, 0.25, 1.0, 1.0, 1.0),
        )

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class PrivCfg(ObsGroup):
        contact_state = ObsTerm(
            func=mdp.contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"])},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        joint_kp_factor = ObsTerm(
            func=mdp.joint_kp_factor,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
            clip=(0.0, 2.0),
            scale=1.0,
        )
        joint_kd_factor = ObsTerm(
            func=mdp.joint_kd_factor,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
            clip=(0.0, 2.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    priv: PrivCfg = PrivCfg()
    scanner: None = None


@configclass
class CommandGatedEventCfg(EventCfg):
    """Domain randomization close to the Genesis WheelLeggedEnv first port."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.6),
            "dynamic_friction_range": (0.2, 1.6),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.5, 1.5),
            "operation": "add",
        },
    )
    add_other_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!base_link$).*"),
            "mass_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )
    add_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    add_other_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!base_link$).*"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True),
            "position_range": (-0.03, 0.03),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class CommandGatedRewardsCfg:
    tracking_lin_x_vel = RewTerm(
        func=mdp.tracking_lin_x_vel_exp,
        weight=1.0,
        params={"command_name": COMMAND_NAME, "sigma": 0.3},
    )
    tracking_lin_y_vel = RewTerm(
        func=mdp.tracking_lin_y_vel_exp,
        weight=0.0,
        params={"command_name": COMMAND_NAME, "sigma": 1.0},
    )
    tracking_ang_vel = RewTerm(
        func=mdp.tracking_ang_vel_exp,
        weight=0.5,
        params={"command_name": COMMAND_NAME, "sigma": 0.25},
    )
    tracking_leg_length = RewTerm(
        func=mdp.tracking_leg_length_thigh_l2,
        weight=-0.2,
        params={
            "command_name": COMMAND_NAME,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_left_leg_2", "joint_right_leg_2"], preserve_order=True),
        },
    )
    projected_gravity = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    tsk = RewTerm(
        func=mdp.tsk_hip_tracking_l2,
        weight=-0.2,
        params={
            "command_name": COMMAND_NAME,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_left_leg_1", "joint_right_leg_1"], preserve_order=True),
            "opposite_sign": False,
        },
    )
    similar_calf = RewTerm(
        func=mdp.similar_calf_l2,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_left_leg_3", "joint_right_leg_3"], preserve_order=True)},
    )
    joint_action_rate = RewTerm(func=mdp.action_rate_l2_slice, weight=-0.01, params={"start": 0, "end": 6})
    wheel_action_rate = RewTerm(func=mdp.action_rate_l2_slice, weight=-0.001, params={"start": 6, "end": 8})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_force = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_3"]), "threshold": 1.0},
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance_range_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_leg_4", "right_leg_4"], preserve_order=True),
            "min_distance": 0.3,
            "max_distance": 0.6,
        },
    )
    survive = RewTerm(func=mdp.is_alive, weight=0.0)
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["joint_left_leg_(1|2|3)", "joint_right_leg_(1|2|3)"]],
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.35},
    )


@configclass
class CommandGatedTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    roll_pitch = DoneTerm(
        func=mdp.roll_pitch_exceeded,
        params={"roll_limit": math.radians(60.0), "pitch_limit": math.radians(60.0)},
    )
    undesired_contacts = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


@configclass
class TitaCommandGatedFlatEnvCfg(TitaFlatEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: CommandGatedObservationsCfg = CommandGatedObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandGatedCommandsCfg = CommandGatedCommandsCfg()
    rewards: CommandGatedRewardsCfg = CommandGatedRewardsCfg()
    terminations: CommandGatedTerminationsCfg = CommandGatedTerminationsCfg()
    events: CommandGatedEventCfg = CommandGatedEventCfg()
    costs: CostsCfg = CostsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 20.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.scanner = None
        self.curriculum.terrain_levels = None
        self.observations.policy.history_length = 10
        self.observations.policy.flatten_history_dim = False
        self.observations.policy.enable_corruption = True
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        }
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class TitaCommandGatedFlatEnvCfg_PLAY(TitaCommandGatedFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_other_com = None
        self.events.add_base_mass = None
        self.events.add_other_mass = None
        self.events.randomize_actuator_gains = None
