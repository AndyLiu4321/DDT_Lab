"""Opt-in Height task using the evaluated PACE candidate from 2026-09-17.

Physics step 1 ms assumes the documented 200 Hz fit command. The historical
checkpoint did not save physics dt; replay validation is still required.
"""
import json
from pathlib import Path

from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.utils import configclass

from ..height_env_cfg import AndyHeightFlatEnvCfg
from .actuator_cfg import PaceDCMotorCfg
from ..rough_env_cfg import ANDY_POSITION_JOINTS, ANDY_DRIVE_JOINTS


CANDIDATE = json.loads((Path(__file__).parent / 'data' / '200hz_candidate.json').read_text())
PARAMS = CANDIDATE['parameters']


class EncoderLimitedPositionAction(JointPositionAction):
    """Targets are encoder angles; articulation limits are physical angles."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        import torch
        self._bias = torch.tensor(
            [PARAMS['encoder_bias'][name] for name in self._joint_names],
            device=self.device,
        )

    def process_actions(self, actions):
        super().process_actions(actions)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._processed_actions.clamp_(
            min=limits[..., 0] - self._bias, max=limits[..., 1] - self._bias,
        )


def apply_pace_height(cfg):
    """Apply after the original Height __post_init__; do not change its defaults."""
    # Preserve 50 Hz policy updates while evaluating the motor at 1 kHz.
    policy_dt = cfg.sim.dt * cfg.decimation
    cfg.sim.dt = 0.001
    cfg.decimation = round(policy_dt / cfg.sim.dt)
    if abs(cfg.decimation * cfg.sim.dt - policy_dt) > 1e-9:
        raise ValueError('Policy interval must be an integer number of 1 ms physics steps')
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.contact_forces.update_period = cfg.sim.dt

    # Candidate applies zero torque-delay steps. Nonzero candidates need the
    # original physics dt to convert delay duration and are not accepted here.
    if PARAMS['delay_steps_applied'] != 0:
        raise ValueError('This configuration supports the selected zero-delay candidate only')
    for group, names, kp, kd in (
        ('legs', ANDY_POSITION_JOINTS, 40.0, 1.0),
        ('drive_wheels', ANDY_DRIVE_JOINTS, 0.0, 0.5),
    ):
        select = lambda key: {name: PARAMS[key][name] for name in names}
        cfg.scene.robot.actuators[group] = PaceDCMotorCfg(
            joint_names_expr=names,
            saturation_effort=11.0, effort_limit=11.0, effort_limit_sim=11.0,
            velocity_limit=12.57, velocity_limit_sim=12.57,
            stiffness=kp, damping=kd, armature=select('armature'),
            friction=select('friction'), dynamic_friction=select('friction'),
            viscous_friction=select('viscous_friction'),
            encoder_bias=select('encoder_bias'), max_delay=0,
        )

    # q_true = q_encoder + bias. Store the physical nominal pose for resets.
    # Existing relative observations then equal q_true - (q_nominal + bias),
    # exactly q_encoder - q_nominal. Do NOT subtract bias a second time.
    nominal = dict(cfg.scene.robot.init_state.joint_pos)
    for name in ANDY_POSITION_JOINTS:
        cfg.scene.robot.init_state.joint_pos[name] = nominal[name] + PARAMS['encoder_bias'][name]
    # Actions must retain the original encoder-frame nominal offsets.
    for action_name, joint in zip(
        ('joint_pos_0', 'joint_pos_1', 'joint_pos_3', 'joint_pos_4'), ANDY_POSITION_JOINTS,
    ):
        action = getattr(cfg.actions, action_name)
        action.class_type = EncoderLimitedPositionAction
        action.use_default_offset = False
        action.offset = nominal[joint]

    # Fixed nominal dynamics for the initial comparison; separate from future
    # domain-randomized training. Terrain friction is not identified by PACE.
    for name in ('add_base_mass', 'add_base_inertia', 'add_base_com',
                 'randomize_actuator_gains', 'physics_material',
                 'base_external_force_torque', 'push_robot'):
        setattr(cfg.events, name, None)
    cfg.observations.policy.enable_corruption = False
    cfg.observations.critic.enable_corruption = False
    cfg.events.reset_robot_joints.func = mdp.reset_joints_by_offset
    cfg.events.reset_robot_joints.params['position_range'] = (0.0, 0.0)
    cfg.events.reset_robot_joints.params['velocity_range'] = (0.0, 0.0)


@configclass
class AndyPaceHeightFlatEnvCfg(AndyHeightFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_pace_height(self)


@configclass
class AndyPaceHeightFlatEnvCfg_PLAY(AndyPaceHeightFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
