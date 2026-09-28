# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import mdp
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from .terrain_features import grid_height_profile


class JumpCommand(CommandTerm):
    """Jump trigger command with the same FSM shape as ``wheelfoot_flat_jump``.

    Command value is the sampled jump intent for the current segment:
    ``0`` keeps recovery/standing behavior, ``1`` asks the policy to perform a
    jump when the internal trigger and stage machine allow it. This term does
    not apply any external velocity or force.
    """

    cfg: "JumpCommandCfg"

    STAGE_RECOVERY = 0
    STAGE_PREP = 1
    STAGE_TAKEOFF = 2
    STAGE_FLIGHT = 3
    STAGE_LAND = 4

    def __init__(self, cfg: "JumpCommandCfg", env):
        super().__init__(cfg, env)

        self.asset: Articulation = env.scene[cfg.asset_cfg.name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.sensor_cfg.name]

        self.jump_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        self.jump_command = self.jump_cmd  # backwards-compatible alias
        self.jump_stage = torch.full(
            (self.num_envs,), self.STAGE_RECOVERY, dtype=torch.int32, device=self.device
        )
        setattr(env, "jump_stage", self.jump_stage)
        self.was_in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_height = torch.zeros(self.num_envs, device=self.device)
        if isinstance(cfg.sensor_cfg.body_ids, slice):
            num_contact_bodies = len(self.contact_sensor.body_names)
        else:
            num_contact_bodies = len(cfg.sensor_cfg.body_ids)
        self.last_contacts = torch.zeros(self.num_envs, num_contact_bodies, dtype=torch.bool, device=self.device)
        self.contact_filt = torch.zeros_like(self.last_contacts)
        self.recovery_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.s1_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.s3_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.flight_reward_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.trigger_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # World-frame take-off position.  Stairs tasks use it to reward
        # clearing a complete tread instead of learning an in-place hop.
        self.takeoff_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.metrics["jump_cmd"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["jump_stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["has_jumped"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["forward_distance"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.jump_cmd

    @property
    def warmup_complete(self) -> torch.Tensor:
        warmup_steps = int(self.cfg.warmup_iterations * self.cfg.steps_per_iteration)
        return torch.full(
            (self.num_envs,),
            self._env.common_step_counter >= warmup_steps,
            device=self.device,
            dtype=torch.bool,
        )

    def _update_metrics(self):
        self.metrics["jump_cmd"] = self.jump_cmd[:, 0]
        self.metrics["jump_stage"] = self.jump_stage.float()
        self.metrics["has_jumped"] = self.has_jumped.float()
        self.metrics["max_height"] = self.max_height
        self.metrics["forward_distance"] = torch.clamp(
            self.asset.data.root_pos_w[:, 0] - self.takeoff_pos_w[:, 0], min=0.0
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return
        self._reset_jump_state(env_ids, absolute=True)

    def _update_command(self):
        self._update_contact_state()
        self.max_height = torch.maximum(self.max_height, self.asset.data.root_pos_w[:, 2])

        if self.cfg.manual_mode:
            manual_mask = self.jump_cmd[:, 0] > 0.0
            manual_takeoff = manual_mask & (self.jump_stage == self.STAGE_RECOVERY)
            self.takeoff_pos_w[manual_takeoff] = self.asset.data.root_pos_w[manual_takeoff]
            self.jump_stage[manual_takeoff] = self.STAGE_TAKEOFF
            self.jump_stage[~manual_mask] = self.STAGE_RECOVERY
            self._update_jump_timers()
            return

        can_trigger = self.warmup_complete
        if self.cfg.play_mode:
            can_trigger = torch.ones_like(can_trigger, dtype=torch.bool)

        self.jump_stage[
            can_trigger & (self.jump_cmd[:, 0] > 0.0) & (self.jump_stage == self.STAGE_RECOVERY)
        ] = self.STAGE_PREP
        trigger_mask = (
            can_trigger
            & (self.jump_cmd[:, 0] > 0.0)
            & (self._env.episode_length_buf >= self.trigger_step)
            & (self.jump_stage == self.STAGE_PREP)
        )
        self.takeoff_pos_w[trigger_mask] = self.asset.data.root_pos_w[trigger_mask]
        self.jump_stage[trigger_mask] = self.STAGE_TAKEOFF

        self._update_jump_timers()

    def _update_jump_timers(self):
        recovery_mask = (
            self.warmup_complete
            & (self.jump_cmd[:, 0] == 0.0)
            & (self.jump_stage == self.STAGE_RECOVERY)
        )
        self.recovery_timer[recovery_mask] += 1
        self.recovery_timer[~recovery_mask] = 0

        s1_mask = self.jump_stage == self.STAGE_TAKEOFF
        self.s1_timer[s1_mask] += 1
        self.s1_timer[~s1_mask] = 0

        flight_mask = self.jump_stage == self.STAGE_FLIGHT
        self.flight_reward_timer[flight_mask] += 1
        self.flight_reward_timer[~flight_mask] = 0

        s3_mask = self.jump_stage == self.STAGE_LAND
        self.s3_timer[s3_mask] += 1
        self.s3_timer[~s3_mask] = 0

        recovery_timeout_steps = int(self.cfg.recovery_resample_s / self._env.step_dt)
        s1_timeout_steps = int(self.cfg.s1_timeout_s / self._env.step_dt)
        s3_timeout_steps = int(self.cfg.s3_timeout_s / self._env.step_dt)
        timeout_ids = (
            (self.recovery_timer >= recovery_timeout_steps)
            | (self.s1_timer >= s1_timeout_steps)
            | (self.s3_timer >= s3_timeout_steps)
        ).nonzero(as_tuple=False).flatten()
        if len(timeout_ids) > 0:
            self._reset_jump_state(timeout_ids, absolute=False)

    def _update_contact_state(self):
        net_forces = self.contact_sensor.data.net_forces_w[:, self.cfg.sensor_cfg.body_ids, :]
        contacts = net_forces[:, :, 2] > self.cfg.contact_force_threshold
        self.contact_filt = torch.logical_or(contacts, self.last_contacts)
        self.last_contacts = contacts.clone()

        airborne = torch.all(~self.contact_filt, dim=1)
        # Optional take-off validation prevents brief wheel unloading or
        # contact-sensor flicker from being counted as a real jump.  Defaults
        # preserve the original contact-only behavior for existing tasks.
        if self.cfg.takeoff_min_height is not None:
            airborne &= self.asset.data.root_pos_w[:, 2] >= self.cfg.takeoff_min_height
        if self.cfg.takeoff_min_vel_z is not None:
            airborne &= self.asset.data.root_lin_vel_w[:, 2] >= self.cfg.takeoff_min_vel_z
        takeoff_ids = airborne & (self.jump_stage == self.STAGE_TAKEOFF)
        self.was_in_flight[takeoff_ids] = True
        self.jump_stage[takeoff_ids] = self.STAGE_FLIGHT
        landed = torch.any(self.contact_filt, dim=1) & self.was_in_flight
        self.has_jumped[landed] = True
        self.jump_stage[landed] = self.STAGE_LAND

    def _reset_jump_state(self, env_ids: torch.Tensor, absolute: bool):
        if self.cfg.play_mode:
            jump_cmd = torch.ones(len(env_ids), device=self.device)
        else:
            jump_cmd = (torch.rand(len(env_ids), device=self.device) < self.cfg.jump_probability).float()
        self.jump_cmd[env_ids, 0] = jump_cmd
        self.jump_stage[env_ids] = self.STAGE_RECOVERY
        self.was_in_flight[env_ids] = False
        self.has_jumped[env_ids] = False
        self.max_height[env_ids] = 0.0
        self.takeoff_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids]
        self.last_contacts[env_ids] = False
        self.contact_filt[env_ids] = False
        self.recovery_timer[env_ids] = 0
        self.s1_timer[env_ids] = 0
        self.s3_timer[env_ids] = 0
        self.flight_reward_timer[env_ids] = 0

        if self.cfg.play_mode:
            delay = torch.full((len(env_ids),), self.cfg.play_trigger_steps, device=self.device, dtype=torch.long)
        else:
            delay = torch.randint(
                self.cfg.trigger_step_range[0],
                self.cfg.trigger_step_range[1],
                (len(env_ids),),
                device=self.device,
                dtype=torch.long,
            )
        base_step = 0 if absolute else self._env.episode_length_buf[env_ids]
        self.trigger_step[env_ids] = base_step + delay


@configclass
class JumpCommandCfg(CommandTermCfg):
    class_type: type = JumpCommand

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*_leg_4"])
    trigger_step_range: tuple[int, int] = (125, 126)
    warmup_iterations: int = 3000
    steps_per_iteration: int = 24
    s1_timeout_s: float = 5.0
    s3_timeout_s: float = 3.0
    recovery_resample_s: float = 3.0
    flight_reward_window_s: float = 0.35
    target_height: float = 0.8
    jump_probability: float = 0.2
    contact_force_threshold: float = 1.0
    takeoff_min_height: float | None = None
    takeoff_min_vel_z: float | None = None
    play_mode: bool = False
    play_trigger_steps: int = 250
    manual_mode: bool = False


class TerrainJumpCommand(JumpCommand):
    """Privileged reward FSM triggered by terrain geometry instead of a command.

    This term remains inside the training environment solely to gate PREP,
    TAKEOFF, FLIGHT and LAND rewards.  The Actor does not observe its command
    value or stage; it must infer when to jump from ``terrain_height_profile``.
    """

    cfg: "TerrainJumpCommandCfg"

    def __init__(self, cfg: "TerrainJumpCommandCfg", env):
        super().__init__(cfg, env)
        self.terrain_sensor = env.scene[cfg.terrain_sensor_cfg.name]
        self.obstacle_distance = torch.full(
            (self.num_envs,), cfg.max_detection_distance, device=self.device
        )
        self.obstacle_height = torch.zeros(self.num_envs, device=self.device)
        self.obstacle_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prep_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # Do not expose the inherited command-intent metric: this FSM has no
        # operator command and the value is only an internal reward activation.
        self.metrics.pop("jump_cmd", None)
        self.metrics["obstacle_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["obstacle_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["obstacle_detected"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        self.metrics["jump_stage"] = self.jump_stage.float()
        self.metrics["has_jumped"] = self.has_jumped.float()
        self.metrics["max_height"] = self.max_height
        self.metrics["forward_distance"] = torch.clamp(
            self.asset.data.root_pos_w[:, 0] - self.takeoff_pos_w[:, 0], min=0.0
        )
        self.metrics["obstacle_distance"] = self.obstacle_distance
        self.metrics["obstacle_height"] = self.obstacle_height
        self.metrics["obstacle_detected"] = self.obstacle_detected.float()

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) > 0:
            self._reset_jump_state(env_ids, absolute=True)

    def _update_command(self):
        self._update_contact_state()
        self._update_obstacle_geometry()
        self.max_height = torch.maximum(self.max_height, self.asset.data.root_pos_w[:, 2])

        recovery = self.jump_stage == self.STAGE_RECOVERY
        begin_prep = recovery & self.obstacle_detected
        self.jump_cmd[begin_prep, 0] = 1.0
        self.jump_stage[begin_prep] = self.STAGE_PREP

        prep = self.jump_stage == self.STAGE_PREP
        self.prep_timer[prep] += 1
        self.prep_timer[~prep] = 0
        prep_timeout_steps = int(self.cfg.prep_timeout_s / self._env.step_dt)
        takeoff = prep & (
            (self.obstacle_distance <= self.cfg.takeoff_distance)
            | (self.prep_timer >= prep_timeout_steps)
        )
        self.takeoff_pos_w[takeoff] = self.asset.data.root_pos_w[takeoff]
        self.jump_stage[takeoff] = self.STAGE_TAKEOFF

        # If the robot turns away before reaching the obstacle, cancel the
        # hidden reward phase instead of rewarding an unrelated jump.
        lost_obstacle = prep & ~self.obstacle_detected
        if torch.any(lost_obstacle):
            lost_ids = lost_obstacle.nonzero(as_tuple=False).flatten()
            self._reset_jump_state(lost_ids, absolute=False)

        self._update_jump_timers()

    def _update_obstacle_geometry(self):
        profile = grid_height_profile(
            self.terrain_sensor.data.ray_hits_w,
            num_longitudinal=self.cfg.num_longitudinal,
            num_lateral=self.cfg.num_lateral,
            reference_start_column=self.cfg.reference_start_column,
            reference_end_column=self.cfg.reference_end_column,
        )
        forward = profile[:, self.cfg.forward_start_column :]
        distances = (
            torch.arange(forward.shape[1], device=self.device, dtype=forward.dtype)
            * self.cfg.grid_resolution
        )
        candidates = (
            (forward >= self.cfg.min_obstacle_height)
            & (forward <= self.cfg.max_obstacle_height)
            & (distances.unsqueeze(0) >= self.cfg.min_detection_distance)
            & (distances.unsqueeze(0) <= self.cfg.max_detection_distance)
        )
        detected = torch.any(candidates, dim=1)
        first_index = torch.argmax(candidates.to(torch.int64), dim=1)
        batch = torch.arange(self.num_envs, device=self.device)
        self.obstacle_detected = detected
        self.obstacle_distance = torch.where(
            detected,
            distances[first_index],
            torch.full_like(self.obstacle_distance, self.cfg.max_detection_distance),
        )
        self.obstacle_height = torch.where(
            detected,
            forward[batch, first_index],
            torch.zeros_like(self.obstacle_height),
        )

    def _reset_jump_state(self, env_ids: torch.Tensor, absolute: bool):
        super()._reset_jump_state(env_ids, absolute)
        self.jump_cmd[env_ids, 0] = 0.0
        if hasattr(self, "prep_timer"):
            self.prep_timer[env_ids] = 0


@configclass
class TerrainJumpCommandCfg(JumpCommandCfg):
    """Configuration for scanner-triggered privileged jump reward stages."""

    class_type: type = TerrainJumpCommand
    terrain_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner")
    num_longitudinal: int = 17
    num_lateral: int = 11
    reference_start_column: int = 6
    reference_end_column: int = 9
    forward_start_column: int = 8
    grid_resolution: float = 0.1
    min_obstacle_height: float = 0.025
    max_obstacle_height: float = 0.35
    min_detection_distance: float = 0.2
    max_detection_distance: float = 0.7
    takeoff_distance: float = 0.35
    prep_timeout_s: float = 2.0
    jump_probability: float = 0.0


class WheelLeggedCommand(CommandTerm):
    """Six-dimensional command for Tita wheel-legged locomotion.

    Layout:
    ``[vx, vy, wz, left_leg_length_cmd, right_leg_length_cmd, tsk_cmd]``.
    The leg-length entries intentionally mirror the Genesis first port and
    are consumed as thigh-joint targets by the reward terms.
    """

    cfg: "WheelLeggedCommandCfg"

    def __init__(self, cfg: "WheelLeggedCommandCfg", env):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.asset_cfg.name]
        self.command_buf = torch.zeros(self.num_envs, 6, device=self.device)
        self.metrics["lin_x_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ang_z_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_buf

    def _update_metrics(self):
        self.metrics["lin_x_error"] = torch.abs(self.command_buf[:, 0] - self.asset.data.root_lin_vel_b[:, 0])
        self.metrics["ang_z_error"] = torch.abs(self.command_buf[:, 2] - self.asset.data.root_ang_vel_b[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        vx_low, vx_high = self._scaled_symmetric_range(self.cfg.lin_vel_x_range, self.cfg.lin_vel_x_curriculum_ratio)
        wz_low, wz_high = self._scaled_symmetric_range(self.cfg.ang_vel_z_range, self.cfg.ang_vel_z_curriculum_ratio)

        self.command_buf[env_ids, 0] = self._uniform(vx_low, vx_high, len(env_ids))
        self.command_buf[env_ids, 1] = self._uniform(*self.cfg.lin_vel_y_range, len(env_ids))

        if self.cfg.high_speed:
            safe_vx = torch.clamp(torch.abs(self.command_buf[env_ids, 0]), min=1.0e-4)
            angv_limit = self.cfg.inverse_linx_angv / safe_vx
            local_wz_low = torch.clamp(torch.full_like(angv_limit, wz_low), min=-angv_limit, max=angv_limit)
            local_wz_high = torch.clamp(torch.full_like(angv_limit, wz_high), min=-angv_limit, max=angv_limit)
            rand = torch.rand(len(env_ids), device=self.device)
            self.command_buf[env_ids, 2] = local_wz_low + rand * (local_wz_high - local_wz_low)

            safe_wz = torch.clamp(torch.abs(self.command_buf[env_ids, 2]), min=1.0e-4)
            tsk_std = self.cfg.inverse_tsk / safe_wz
            tsk = torch.randn(len(env_ids), device=self.device) * tsk_std
            self.command_buf[env_ids, 5] = torch.clamp(tsk, *self.cfg.tsk_range)

            leg_mean = self._uniform(*self.cfg.leg_length_range, len(env_ids))
            leg_std = self.cfg.inverse_leg_length / safe_wz
            left_leg = leg_mean + torch.randn(len(env_ids), device=self.device) * leg_std
            right_leg = leg_mean + torch.randn(len(env_ids), device=self.device) * leg_std
            self.command_buf[env_ids, 3] = torch.clamp(left_leg, *self.cfg.leg_length_range)
            self.command_buf[env_ids, 4] = torch.clamp(right_leg, *self.cfg.leg_length_range)
        else:
            self.command_buf[env_ids, 2] = self._uniform(wz_low, wz_high, len(env_ids))
            self.command_buf[env_ids, 3] = self._uniform(*self.cfg.leg_length_range, len(env_ids))
            self.command_buf[env_ids, 4] = self._uniform(*self.cfg.leg_length_range, len(env_ids))
            self.command_buf[env_ids, 5] = self._uniform(*self.cfg.tsk_range, len(env_ids))

        if self.cfg.zero_stable:
            stop_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.zero_command_probability
            if torch.any(stop_mask):
                self.command_buf[env_ids[stop_mask], :3] = 0.0

    def _update_command(self):
        # First version keeps fixed sampling ranges. Curriculum range updates can
        # be added here or via a CurriculumTerm using the logged tracking errors.
        pass

    def _uniform(self, low: float, high: float, n: int) -> torch.Tensor:
        return torch.empty(n, device=self.device).uniform_(low, high)

    @staticmethod
    def _scaled_symmetric_range(full_range: tuple[float, float], ratio: float) -> tuple[float, float]:
        if ratio >= 1.0:
            return full_range
        low, high = full_range
        if low < 0.0 < high:
            return low * ratio, high * ratio
        return low, low + (high - low) * ratio


@configclass
class WheelLeggedCommandCfg(CommandTermCfg):
    class_type: type = WheelLeggedCommand

    resampling_time_range: tuple[float, float] = (3.0, 3.0)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    lin_vel_x_range: tuple[float, float] = (-2.0, 2.0)
    lin_vel_y_range: tuple[float, float] = (0.0, 0.0)
    ang_vel_z_range: tuple[float, float] = (-12.0, 12.0)
    leg_length_range: tuple[float, float] = (0.0, 1.0)
    tsk_range: tuple[float, float] = (-0.3, 0.3)
    high_speed: bool = True
    inverse_linx_angv: float = 1.0
    inverse_tsk: float = 2.0
    inverse_leg_length: float = 2.0
    zero_stable: bool = True
    zero_command_probability: float = 0.02
    lin_vel_x_curriculum_ratio: float = 0.3
    ang_vel_z_curriculum_ratio: float = 0.05
class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects pit-like terrains (see
    :attr:`UniformThresholdVelocityCommandCfg.pit_terrain_names`) and applies restrictions:
    - For pit-like terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on any pit-like terrain (real-time check every step)
        on_pits = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for terrain_name in self.cfg.pit_terrain_names:
            on_pits |= is_robot_on_terrain(self._env, terrain_name)

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            if self.cfg.heading_command:
                # Actively steer toward world heading 0 (the pit-crossing direction) every step
                # instead of just zeroing the yaw command, so a robot that enters/spawns on the
                # pit off-heading actually turns to align with the crossing direction.
                self.heading_target[pit_env_ids] = 0.0
                heading_error = math_utils.wrap_to_pi(
                    self.heading_target[pit_env_ids] - self.robot.data.heading_w[pit_env_ids]
                )
                self.vel_command_b[pit_env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )
            else:
                self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand

    pit_terrain_names: tuple[str, ...] = ("pits", "rails", "boxes")
    """Sub-terrain names (keys in ``TerrainGeneratorCfg.sub_terrains``) that should restrict the
    velocity command to forward-only movement (see :meth:`UniformThresholdVelocityCommand._update_command`)."""


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
