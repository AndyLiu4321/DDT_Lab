"""Zero-command fine tuning; does not alter the exported observation/action ABI."""
import torch
from ..height_mdp import height_error
from .robust_mdp import RobustHeightCommand


def stationary_mask(env, command_name='base_velocity'):
    command = env.command_manager.get_command(command_name)
    return (command[:, 0].abs() < 0.05) & (command[:, 2].abs() < 0.05)


def stationary_translation(env, command_name='base_velocity'):
    speed = env.scene['robot'].data.root_lin_vel_b[:, :2].norm(dim=1)
    return stationary_mask(env, command_name) * (speed - 0.01).clamp_min(0).square()


def stationary_yaw(env, command_name='base_velocity'):
    rate = env.scene['robot'].data.root_ang_vel_b[:, 2].abs()
    return stationary_mask(env, command_name) * (rate - 0.02).clamp_min(0).square()


def stationary_tracking(env, command_name='base_velocity'):
    robot = env.scene['robot']
    error = robot.data.root_lin_vel_b[:, :2].square().sum(dim=1) / 0.05**2
    error += robot.data.root_ang_vel_b[:, 2].square() / 0.10**2
    return stationary_mask(env, command_name) * torch.exp(-error)


def stationary_leg_motion(env, asset_cfg, command_name='base_velocity'):
    # Only leg settling is gated by height; base translation/yaw are not.
    settled = torch.exp(-height_error(env, command_name).square() / 0.0025)
    legs = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return stationary_mask(env, command_name) * settled * legs.square().sum(dim=1)


class StationaryRobustHeightCommand(RobustHeightCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._still_count = torch.zeros(self.num_envs, device=self.device)
        self._still_speed = torch.zeros_like(self._still_count)
        self._still_yaw = torch.zeros_like(self._still_count)
        self._still_ok = torch.zeros_like(self._still_count)
        for name in ('stationary_speed_mps', 'stationary_yaw_radps', 'stationary_fraction_ok',
                     'stationary_seconds'):
            self.metrics[name] = torch.zeros_like(self._still_count)

    def _update_metrics(self):
        super()._update_metrics()
        mask = ((self.command[:, 0].abs() < .05) & (self.command[:, 2].abs() < .05)).float()
        robot = self._env.scene['robot']
        speed = robot.data.root_lin_vel_b[:, :2].norm(dim=1)
        yaw = robot.data.root_ang_vel_b[:, 2].abs()
        self._still_count += mask
        self._still_speed += mask * speed
        self._still_yaw += mask * yaw
        self._still_ok += mask * ((speed < .03) & (yaw < .05)).float()
        count = self._still_count.clamp_min(1)
        self.metrics['stationary_speed_mps'][:] = self._still_speed / count
        self.metrics['stationary_yaw_radps'][:] = self._still_yaw / count
        self.metrics['stationary_fraction_ok'][:] = self._still_ok / count
        self.metrics['stationary_seconds'][:] = self._still_count * self._env.step_dt

    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for value in (self._still_count, self._still_speed, self._still_yaw, self._still_ok):
            value[ids] = 0
        return result
