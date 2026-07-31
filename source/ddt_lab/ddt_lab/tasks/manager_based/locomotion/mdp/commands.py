# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass


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

        self.metrics["jump_cmd"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["jump_stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["has_jumped"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_height"] = torch.zeros(self.num_envs, device=self.device)

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
            self.jump_stage[manual_mask & (self.jump_stage == self.STAGE_RECOVERY)] = self.STAGE_TAKEOFF
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
    play_mode: bool = False
    play_trigger_steps: int = 250
    manual_mode: bool = False


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
