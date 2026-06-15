# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tita fall-recovery env — CartPole-style survival training (flat terrain, NP3O).

Design mirrors locomotion_env_cfg.py (CartPole):

  CartPole                          Tita Recovery
  ─────────────────────────────     ──────────────────────────────────
  alive          = +1.0 / step      alive          = +1.0 / step
  terminating    = -2.0 on fall     is_terminated  = -2.0 on base contact
  pole_pos       = -1.0 (task)      flat_orientation_l2 = -5.0
                                    base_height_l2       = -10.0
                                    upward               = +1.0

Key env differences from the standard flat env:
* ``base_contact`` termination is KEPT — falling flat ends the episode, just
  like CartPole terminates when the cart escapes bounds.  Without this,
  ``is_alive`` / ``is_terminated`` are constant and provide no gradient.
* ``reset_base`` samples roll / pitch up to ±60° so the robot regularly starts
  tilted and must self-right before falling further.
* Velocity-command rewards are suppressed during recovery training — the robot
  cannot follow speed commands while learning to stand.
* ``CostsCfg`` (inherited) activates NP3O Lagrangian constraints to protect
  hardware during violent self-righting attempts.
"""

from isaaclab.utils import configclass

from .flat_env_cfg import TitaFlatEnvCfg

_60_DEG = 1.047  # radians


@configclass
class TitaRecoveryEnvCfg(TitaFlatEnvCfg):
    """Tita flat-terrain fall-recovery (CartPole-style survival training)."""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------ #
        # Termination: KEEP base_contact — falling flat ends the episode      #
        # (same role as cart_out_of_bounds in CartPole).  Without this,       #
        # is_alive / is_terminated are always constant and useless.           #
        # ------------------------------------------------------------------ #
        # (base_contact is inherited from TitaRoughEnvCfg — no change needed)

        # ------------------------------------------------------------------ #
        # Reset: ±60° tilt gives the robot a moderate initial challenge.      #
        # Starts fully fallen (±180°) would immediately trigger base_contact  #
        # before the policy can act.                                          #
        # ------------------------------------------------------------------ #
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.1, 0.5),
                "roll": (-_60_DEG, _60_DEG),
                "pitch": (-_60_DEG, _60_DEG),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        # ------------------------------------------------------------------ #
        # Rewards — CartPole pattern                                          #
        # ------------------------------------------------------------------ #
        # Survival signal (CartPole: alive +1, terminating -2)
        self.rewards.alive.weight = 1.0
        self.rewards.is_terminated.weight = -2.0

        # Recovery / orientation signal
        self.rewards.upward.weight = 1.0           # dense: 4 upright → 0 inverted
        self.rewards.flat_orientation_l2.weight = -5.0   # unchanged (strong)

        # Height: must be strong enough to prevent the "correct orientation
        # but lying flat on wheels" local optimum.
        self.rewards.base_height_l2.weight = -10.0

        # Suppress velocity commands — impossible to track while learning to
        # stand, and the conflicting signal slows recovery learning.
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0


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
