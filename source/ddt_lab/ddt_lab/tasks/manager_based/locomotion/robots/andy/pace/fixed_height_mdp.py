"""Fixed 0.35 m target with explicit stationary-state training."""
import torch
from .stationary_mdp import StationaryRobustHeightCommand, stationary_mask


class FixedHeightCommand(StationaryRobustHeightCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._command[env_ids, 1] = .35

    def _update_command(self):
        # Reserve the old observation slot for constant height; no height commands.
        self._command[:, 1] = .35


def stationary_pose(env):
    robot = env.scene['robot']
    tilt = robot.data.projected_gravity_b[:, :2].square().sum(-1)
    height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return stationary_mask(env) * torch.exp(-tilt / .04 - (height - .35).square() / .0009)
