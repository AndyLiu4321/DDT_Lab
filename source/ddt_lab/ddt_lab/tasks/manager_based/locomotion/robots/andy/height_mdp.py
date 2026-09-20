"""Height-command MDP adapted from the local AndyMini reference (flat ground)."""

import torch
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass


class HeightCommand(CommandTerm):
    """Sample [forward velocity (m/s), base height (m), yaw rate (rad/s)]."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._command[:, 1] = cfg.base_height[0]
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        count = len(env_ids)
        samples = torch.empty(count, 3, device=self.device)
        samples[:, 0].uniform_(*self.cfg.lin_vel_x)
        samples[:, 1].uniform_(*self.cfg.base_height)
        samples[:, 2].uniform_(*self.cfg.ang_vel_z)
        selector = torch.rand(count, device=self.device)
        # Reference: 10% extreme low, 15% minimum, 25% maximum, 50% uniform.
        samples[selector < 0.10, 1] = self.cfg.extreme_low_height
        samples[(selector >= 0.10) & (selector < 0.25), 1] = self.cfg.base_height[0]
        samples[(selector >= 0.25) & (selector < 0.50), 1] = self.cfg.base_height[1]
        samples[:, 0] *= samples[:, 0].abs() > 0.2
        # Explicitly train stationary height control; continuous yaw sampling
        # otherwise almost never produces the zero-motion commands used in play.
        standing = torch.rand(count, device=self.device) < self.cfg.standing_fraction
        samples[standing, 0] = 0.0
        samples[standing, 2] = 0.0
        self._command[env_ids] = samples

    def _update_command(self):
        pass

    def _update_metrics(self):
        steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["height_error"] += (base_height(self._env) - self.command[:, 1]).abs() / steps


@configclass
class HeightCommandCfg(CommandTermCfg):
    class_type: type = HeightCommand
    resampling_time_range: tuple = (10.0, 10.0)
    lin_vel_x: tuple = (-0.6, 0.6)
    base_height: tuple = (0.22, 0.37)
    ang_vel_z: tuple = (-0.6, 0.6)
    extreme_low_height: float = 0.22
    standing_fraction: float = 0.2


class LimitedJointPositionAction(JointPositionAction):
    """Clamp PD position targets to the articulation limits, as in AndyMini."""

    def process_actions(self, actions):
        super().process_actions(actions)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._processed_actions.clamp_(min=limits[..., 0], max=limits[..., 1])


def base_height(env):
    return env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def height_error(env, command_name="base_velocity", tolerance=0.0):
    """Absolute height error outside a symmetric tolerance band (meters)."""
    error = (base_height(env) - env.command_manager.get_command(command_name)[:, 1]).abs()
    return (error - tolerance).clamp_min(0.0)


def tracking_height(env, command_name="base_velocity", sigma=0.0025, tolerance=0.0):
    return torch.exp(-height_error(env, command_name, tolerance).square() / sigma)


def tracking_forward(env, command_name="base_velocity", sigma=0.25):
    error = env.command_manager.get_command(command_name)[:, 0] - env.scene["robot"].data.root_lin_vel_b[:, 0]
    return torch.exp(-error.square() / sigma)


def auxiliary_contact(env, sensor_cfg: SceneEntityCfg, low: bool, command_name="base_velocity"):
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    fraction = (forces.norm(dim=-1) > 1.0).float().mean(dim=1)
    low_mode = env.command_manager.get_command(command_name)[:, 1] <= 0.22
    return fraction * (low_mode if low else ~low_mode)


def tracking_leg_height(env, leg_cfg: SceneEntityCfg, position_cfg: SceneEntityCfg, command_name="base_velocity"):
    term = env.command_manager.get_term(command_name)
    height = term.command[:, 1]
    lower, upper = term.cfg.base_height
    ratio = ((height - lower) / max(upper - lower, 1e-6)).clamp(0.0, 1.0)
    target = 0.05 + ratio * (1.18 - 0.05)
    positions = env.scene["robot"].data.joint_pos
    error = (positions[:, leg_cfg.joint_ids] - target[:, None]).square().mean(dim=1)
    low_error = positions[:, position_cfg.joint_ids].square().mean(dim=1)
    return torch.exp(-torch.where(height <= 0.22, low_error, error) / 0.04)


def below_min_height(env, minimum=0.12):
    return base_height(env) < minimum


def stationary_motion(env, asset_cfg: SceneEntityCfg, command_name="base_velocity"):
    """Penalize settled leg motion at zero vx/yaw, without locking height changes."""
    command = env.command_manager.get_command(command_name)
    stationary = (command[:, 0].abs() < 0.05) & (command[:, 2].abs() < 0.05)
    # Fade in as the robot reaches its target; permit crouching/extension first.
    settled = torch.exp(-height_error(env, command_name).square() / 0.0025)
    robot = env.scene[asset_cfg.name]
    leg_speed = robot.data.joint_vel[:, asset_cfg.joint_ids].square().sum(dim=1)
    base_motion = robot.data.root_lin_vel_b[:, :2].square().sum(dim=1)
    yaw_motion = robot.data.root_ang_vel_b[:, 2].square()
    return stationary * settled * (leg_speed + base_motion + yaw_motion)
