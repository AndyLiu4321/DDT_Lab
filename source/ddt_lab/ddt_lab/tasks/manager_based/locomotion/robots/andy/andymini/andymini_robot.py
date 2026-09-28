import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.b2w.b2w_robot import B2w
from legged_gym.utils.math import get_scale_shift

from .andymini_config import AndyMiniRoughCfg


class AndyMini(B2w):
    """AndyMini with six actuators and two passive auxiliary-wheel DOFs."""

    cfg: AndyMiniRoughCfg

    controlled_dof_names = (
        "joint_left_leg_1",
        "joint_left_leg_2",
        "joint_left_leg_3",
        "joint_right_leg_1",
        "joint_right_leg_2",
        "joint_right_leg_3",
    )
    auxiliary_dof_names = (
        "joint_left_wheel_1",
        "joint_right_wheel_1",
    )
    driven_wheel_dof_names = (
        "joint_left_leg_3",
        "joint_right_leg_3",
    )

    def step(self, actions):
        """Apply actions while keeping history tied to the task input size."""
        self.obs_hist_buf = self.obs_hist_buf[:, self.num_obs :]
        self.obs_hist_buf = torch.cat((self.obs_hist_buf, self.obs_buf), dim=-1)
        self.prev_privileged_obs_buf = self.privileged_obs_buf

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        # 每个环境独立采样延迟子步数；延迟期间继续执行上一个控制周期的动作。
        if self.cfg.domain_rand.randomize_action_latency:
            latency_min, latency_max = self.cfg.domain_rand.latency_range
            action_latency = torch.randint(
                int(latency_min),
                int(latency_max) + 1,
                (self.num_envs, 1),
                device=self.device,
            )
        else:
            action_latency = torch.zeros(
                self.num_envs, 1, dtype=torch.long, device=self.device
            )

        for substep in range(self.cfg.control.decimation):
            delayed = substep < action_latency
            applied_actions = torch.where(delayed, self.last_actions, self.actions)
            self.torques = self._compute_torques(applied_actions).view(
                self.torques.shape
            )
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.prev_privileged_obs_buf,
            self.obs_hist_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def _get_noise_scale_vec(self, cfg):
        """Build noise scales for the 33-element proprioceptive input."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        num_actions = self.num_actions
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0
        noise_vec[9 : 9 + num_actions] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[9 + num_actions : 9 + 2 * num_actions] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[9 + 2 * num_actions : 9 + 3 * num_actions] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        return noise_vec

    def _create_envs(self):
        super()._create_envs()

        missing = set(self.controlled_dof_names + self.auxiliary_dof_names) - set(
            self.dof_names
        )
        if missing:
            raise ValueError(f"AndyMini URDF is missing DOFs: {sorted(missing)}")

        name_to_index = {name: i for i, name in enumerate(self.dof_names)}
        self.controlled_dof_indices = torch.tensor(
            [name_to_index[name] for name in self.controlled_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        self.auxiliary_dof_indices = torch.tensor(
            [name_to_index[name] for name in self.auxiliary_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        self.driven_wheel_dof_indices = torch.tensor(
            [name_to_index[name] for name in self.driven_wheel_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        position_controlled_dof_names = [
            name
            for name in self.controlled_dof_names
            if name not in self.driven_wheel_dof_names
        ]
        self.position_controlled_dof_indices = torch.tensor(
            [name_to_index[name] for name in position_controlled_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        self.extreme_low_position_targets = torch.tensor(
            [
                self.cfg.init_state.extreme_low_joint_angles[name]
                for name in position_controlled_dof_names
            ],
            dtype=torch.float,
            device=self.device,
        )
        controlled_name_to_index = {
            name: i for i, name in enumerate(self.controlled_dof_names)
        }
        self.driven_wheel_action_indices = torch.tensor(
            [controlled_name_to_index[name] for name in self.driven_wheel_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        self.leg_2_action_indices = torch.tensor(
            [
                controlled_name_to_index[name]
                for name in self.controlled_dof_names
                if "leg_2" in name
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.leg_2_dof_indices = self.controlled_dof_indices[
            self.leg_2_action_indices
        ]
        self.auxiliary_body_indices = torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], "left_wheel_1"
                ),
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], "right_wheel_1"
                ),
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _process_rigid_body_props(self, props, env_id):
        """创建环境时应用基座质量随机化；可选启用质心随机化。"""
        if self.cfg.domain_rand.randomize_base_mass:
            mass_min, mass_max = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(mass_min, mass_max)

        # Aug26 run 虽然配置标记为 True，但当时的 AndyMini 实现没有把
        # COM 随机量写入 PhysX；复现模式保留这一实际训练行为。
        reproduce_aug26 = getattr(self.cfg, "reproduce_aug26_behavior", False)
        if (
            not reproduce_aug26
            and self.cfg.domain_rand.randomize_com_displacement
        ):
            com_min, com_max = self.cfg.domain_rand.com_displacement_range
            displacement = np.random.uniform(com_min, com_max, size=3)
            original_com = props[0].com
            props[0].com = gymapi.Vec3(
                original_com.x + displacement[0],
                original_com.y + displacement[1],
                original_com.z + displacement[2],
            )
        return props

    def _init_buffers(self):
        # B2w allocates gains and torques using num_actions, although PhysX
        # requires one torque for every physical DOF. Temporarily expose all
        # eight DOFs while it builds physical buffers, then restore six policy
        # actions and replace only the action buffers.
        policy_num_actions = self.num_actions
        self.num_actions = self.num_dof
        super()._init_buffers()
        self.num_actions = policy_num_actions
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device
        )
        self.last_actions = torch.zeros_like(self.actions)
        # 控制域随机张量按物理 DOF 保存，并在训练过程中周期性重采样。
        self.motor_strength_factors = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device
        )
        self.kp_factors = torch.ones_like(self.motor_strength_factors)
        self.kd_factors = torch.ones_like(self.motor_strength_factors)
        self._randomize_control_factors(
            torch.arange(self.num_envs, device=self.device)
        )
        # command = [lin_vel_x, base_height, ang_vel_yaw].
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.height_measurements,
                self.obs_scales.ang_vel,
            ],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # The inherited noise builder uses num_actions to lay out the joint
        # fields, so rebuild it after restoring the six-action policy width.
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    def _randomize_control_factors(self, env_ids):
        """为指定环境重采样电机强度、Kp 和 Kd 系数。"""
        if len(env_ids) == 0:
            return

        # Aug26 run 中这些配置项虽然为 True，但力矩控制器尚未使用随机
        # 系数。保持单位系数才能复现该 checkpoint 对应的训练分布。
        if getattr(self.cfg, "reproduce_aug26_behavior", False):
            self.motor_strength_factors[env_ids] = 1.0
            self.kp_factors[env_ids] = 1.0
            self.kd_factors[env_ids] = 1.0
            return

        count = len(env_ids)
        if self.cfg.domain_rand.randomize_motor_strength:
            lower, upper = self.cfg.domain_rand.motor_strength_range
            self.motor_strength_factors[env_ids] = torch_rand_float(
                lower, upper, (count, 1), device=self.device
            )
        if self.cfg.domain_rand.randomize_Kp_factor:
            lower, upper = self.cfg.domain_rand.Kp_factor_range
            self.kp_factors[env_ids] = torch_rand_float(
                lower, upper, (count, 1), device=self.device
            )
        if self.cfg.domain_rand.randomize_Kd_factor:
            lower, upper = self.cfg.domain_rand.Kd_factor_range
            self.kd_factors[env_ids] = torch_rand_float(
                lower, upper, (count, 1), device=self.device
            )

    def _post_physics_step_callback(self):
        """执行指令更新、地形测量、随机推动和周期性控制参数随机化。"""
        super()._post_physics_step_callback()
        randomization_interval = max(
            1,
            int(round(self.cfg.domain_rand.rand_interval_s / self.dt)),
        )
        env_ids = (
            self.episode_length_buf % randomization_interval == 0
        ).nonzero(as_tuple=False).flatten()
        self._randomize_control_factors(env_ids)

    def _resample_commands(self, env_ids):
        """Sample [forward velocity, base height, yaw velocity, heading]."""
        if len(env_ids) == 0:
            return

        count = len(env_ids)
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (count, 1),
            device=self.device,
        ).squeeze(1)
        height_min = self.command_ranges["base_height"][0]
        height_max = self.command_ranges["base_height"][1]
        height_commands = torch_rand_float(
            height_min,
            height_max,
            (count, 1),
            device=self.device,
        ).squeeze(1)
        # Explicit distribution: 10% at 0.22 m, 15% at 0.23 m, 25% at
        # 0.37 m, and 50% continuously sampled from [0.23, 0.37] m.
        height_selector = torch.rand(count, device=self.device)
        height_commands = torch.where(
            height_selector < 0.10,
            torch.full_like(
                height_commands, self.cfg.commands.extreme_low_height
            ),
            height_commands,
        )
        height_commands = torch.where(
            (height_selector >= 0.10) & (height_selector < 0.25),
            torch.full_like(height_commands, height_min),
            height_commands,
        )
        height_commands = torch.where(
            (height_selector >= 0.25) & (height_selector < 0.50),
            torch.full_like(height_commands, height_max),
            height_commands,
        )
        self.commands[env_ids, 1] = height_commands
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (count, 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (count, 1),
                device=self.device,
            ).squeeze(1)

        # Only the forward-speed command is dead-banded. The leg_2 command
        # must never be cleared to zero.
        self.commands[env_ids, 0] *= (
            torch.abs(self.commands[env_ids, 0]) > 0.2
        )
        # Aug26 run 只对前进速度使用死区；关闭复现模式后才给偏航指令
        # 添加死区，以覆盖“双零指令”静止场景。
        if not getattr(self.cfg, "reproduce_aug26_behavior", False):
            self.commands[env_ids, 2] *= (
                torch.abs(self.commands[env_ids, 2]) > 0.2
            )

    def check_termination(self):
        """Use a clearance threshold below the minimum 0.22 m command."""
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

        base_clearance = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
            dim=1,
        )
        self.base_contact_buf = base_clearance < self.cfg.rewards.min_base_height
        self.reset_buf |= self.base_contact_buf

    def _update_terrain_curriculum(self, env_ids):
        """Update terrain levels using forward speed, not the height command."""
        if not self.init_done:
            return

        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        move_up = distance > self.terrain.env_length / 2
        expected_distance = (
            torch.abs(self.commands[env_ids, 0])
            * self.max_episode_length_s
            * 0.5
        )
        move_down = (distance < expected_distance) & ~move_up
        self.terrain_levels[env_ids] += move_up.long() - move_down.long()
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], min=0),
        )
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def _compute_torques(self, actions):
        """Map six policy actions to eight DOFs and leave casters unpowered."""
        actions_scaled = torch.zeros_like(self.dof_pos)
        actions_scaled[:, self.controlled_dof_indices] = (
            actions * self.cfg.control.action_scale
        )
        actions_scaled[:, self.leg_2_dof_indices] = (
            actions[:, self.leg_2_action_indices]
            * self.cfg.control.leg_2_action_scale
        )
        actions_scaled[:, self.driven_wheel_dof_indices] = 0.0

        # Clamp only position-controlled joints. Continuous drive and passive
        # wheels do not have meaningful position limits.
        position_targets = self.default_dof_pos + actions_scaled
        position_indices = self.position_controlled_dof_indices
        position_margin = self.cfg.control.position_limit_margin
        position_lower = self.dof_pos_limits[position_indices, 0] + position_margin
        position_upper = self.dof_pos_limits[position_indices, 1] - position_margin
        position_targets[:, position_indices] = torch.maximum(
            torch.minimum(
                position_targets[:, position_indices], position_upper
            ),
            position_lower,
        )
        # Expose the actual PD targets for play_height diagnostics.
        self.position_targets = position_targets

        velocity_target = torch.zeros_like(self.dof_vel)
        velocity_target[:, self.driven_wheel_dof_indices] = (
            actions[:, self.driven_wheel_action_indices] * self.cfg.control.vel_scale
        )
        # 保存最后一个物理子步实际使用的速度目标，供回放诊断和 CSV 记录。
        self.velocity_targets = velocity_target

        if self.cfg.control.control_type != "P":
            raise NameError("AndyMini's mixed position/velocity controller requires P mode")

        effective_p_gains = self.p_gains * self.kp_factors
        effective_d_gains = self.d_gains * self.kd_factors
        torques = effective_p_gains * (
            position_targets - self.dof_pos
        ) + effective_d_gains * (velocity_target - self.dof_vel)
        # 电机强度系数模拟执行器输出偏差。
        torques *= self.motor_strength_factors
        # Keep this explicit even if an auxiliary-wheel gain is accidentally
        # introduced in the configuration later.
        torques[:, self.auxiliary_dof_indices] = 0.0
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):
        """Build policy observations from actuated joints only."""
        controlled_pos = self.dof_pos[:, self.controlled_dof_indices].clone()
        controlled_vel = self.dof_vel[:, self.controlled_dof_indices]
        controlled_default = self.default_dof_pos[:, self.controlled_dof_indices]
        controlled_err = controlled_pos - controlled_default
        controlled_err[:, self.driven_wheel_action_indices] = 0.0
        controlled_pos[:, self.driven_wheel_action_indices] = 0.0

        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                controlled_err * self.obs_scales.dof_pos,
                controlled_vel * self.obs_scales.dof_vel,
                controlled_pos,
                self.actions,
            ),
            dim=-1,
        )

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
            -1,
            1,
        ) * self.obs_scales.height_measurements
        contact_scale, contact_shift = get_scale_shift(
            self.cfg.normalization.contact_force_range
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.obs_buf,
                self.base_lin_vel * self.obs_scales.lin_vel,
                (self.contact_forces.view(self.num_envs, -1) - contact_shift)
                * contact_scale,
                heights,
            ),
            dim=-1,
        )

    def _reward_dof_vel(self):
        leg_indices = self.controlled_dof_indices[:]
        leg_vel = self.dof_vel[:, leg_indices].clone()
        leg_vel[:, self.driven_wheel_action_indices] = 0.0
        return torch.sum(torch.square(leg_vel), dim=1)

    def _reward_tracking_lin_vel(self):
        """Track forward velocity only; command[1] is base height, not vy."""
        velocity_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-velocity_error / self.cfg.rewards.tracking_sigma)

    def _reward_lin_vel_y(self):
        """惩罚机体坐标系侧向速度，减少侧滑。"""
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_zero_command_motion(self):
        """双零指令时惩罚平面线速度和偏航角速度。"""
        zero_command = (
            (torch.abs(self.commands[:, 0]) < 0.05)
            & (torch.abs(self.commands[:, 2]) < 0.05)
        )
        motion = (
            torch.square(self.base_lin_vel[:, 0])
            + torch.square(self.base_lin_vel[:, 1])
            + torch.square(self.base_ang_vel[:, 2])
        )
        return motion * zero_command.float()

    def _reward_tracking_base_height(self):
        """Smoothly reward the per-environment height command."""
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
            dim=1,
        )
        height_error = torch.square(base_height - self.commands[:, 1])
        return torch.exp(
            -height_error / self.cfg.rewards.height_tracking_sigma
        )

    def _reward_base_height_error(self):
        """Penalize absolute error from the per-environment height command."""
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
            dim=1,
        )
        return torch.abs(base_height - self.commands[:, 1])

    def _auxiliary_contact_fraction(self):
        """Return the fraction of the two passive wheels currently in contact."""
        auxiliary_forces = self.contact_forces[
            :, self.auxiliary_body_indices, :
        ]
        contacts = torch.norm(auxiliary_forces, dim=-1) > (
            self.cfg.rewards.auxiliary_contact_force_threshold
        )
        return torch.mean(contacts.float(), dim=1)

    def _reward_auxiliary_contact_low(self):
        """Reward passive-wheel contact for commands at or below 0.22 m."""
        low_mode = self.commands[:, 1] <= (
            self.cfg.rewards.auxiliary_contact_height_threshold
        )
        return self._auxiliary_contact_fraction() * low_mode.float()

    def _reward_auxiliary_contact_high(self):
        """Penalize passive-wheel contact for commands above 0.22 m."""
        high_mode = self.commands[:, 1] > (
            self.cfg.rewards.auxiliary_contact_height_threshold
        )
        return self._auxiliary_contact_fraction() * high_mode.float()

    def _reward_tracking_leg_2_height(self):
        """Guide height posture, including all-zero leg joints at 0.22 m."""
        threshold = self.cfg.rewards.auxiliary_contact_height_threshold
        height_min = self.command_ranges["base_height"][0]
        height_max = self.command_ranges["base_height"][1]
        high_ratio = torch.clamp(
            (self.commands[:, 1] - height_min)
            / max(height_max - height_min, 1e-6),
            min=0.0,
            max=1.0,
        )
        target = self.cfg.rewards.leg_2_low_target + high_ratio * (
            self.cfg.rewards.leg_2_high_target
            - self.cfg.rewards.leg_2_low_target
        )
        leg_2_pos = self.dof_pos[:, self.leg_2_dof_indices]
        error = torch.mean(
            torch.square(leg_2_pos - target.unsqueeze(1)), dim=1
        )
        extreme_low_mode = self.commands[:, 1] <= threshold
        low_joint_error = torch.mean(
            torch.square(
                self.dof_pos[:, self.position_controlled_dof_indices]
                - self.extreme_low_position_targets
            ),
            dim=1,
        )
        error = torch.where(extreme_low_mode, low_joint_error, error)
        return torch.exp(-error / self.cfg.rewards.leg_2_tracking_sigma)

    # def _reward_joint_symmetry(self):
    #     """Penalize non-mirrored left/right leg configurations.

    #     The right leg axes are opposite to the left axes in the URDF, so a
    #     geometrically symmetric pose satisfies q_left + q_right = 0. The two
    #     continuously rotating drive wheels are intentionally excluded.
    #     """
    #     left_leg = self.dof_pos[:, self.controlled_dof_indices[[0, 1]]]
    #     right_leg = self.dof_pos[:, self.controlled_dof_indices[[3, 4]]]
    #     return torch.sum(torch.square(left_leg + right_leg), dim=1)
    def _reward_joint_symmetry(self):
        left_leg = self.dof_pos[:, self.controlled_dof_indices[[0, 1]]]
        right_leg = self.dof_pos[:, self.controlled_dof_indices[[3, 4]]]
        return torch.sum(torch.square(left_leg - right_leg), dim=1)
    def _reward_dof_acc(self):
        current = self.dof_vel[:, self.controlled_dof_indices]
        previous = self.last_dof_vel[:, self.controlled_dof_indices]
        return torch.sum(torch.square((previous - current) / self.dt), dim=1)

    def _reward_dof_pos_limits(self):
        pos = self.dof_pos[:, self.controlled_dof_indices]
        limits = self.dof_pos_limits[self.controlled_dof_indices]
        out_of_limits = -(pos - limits[:, 0]).clip(max=0.0)
        out_of_limits += (pos - limits[:, 1]).clip(min=0.0)
        out_of_limits[:, self.driven_wheel_action_indices] = 0.0
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        vel = self.dof_vel[:, self.controlled_dof_indices]
        limits = self.dof_vel_limits[self.controlled_dof_indices]
        return torch.sum(
            (
                torch.abs(vel)
                - limits * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_stand_still(self):
        dof_err = (
            self.dof_pos[:, self.controlled_dof_indices]
            - self.default_dof_pos[:, self.controlled_dof_indices]
        )
        dof_err[:, self.driven_wheel_action_indices] = 0.0
        return torch.sum(torch.abs(dof_err), dim=1) * (
            torch.abs(self.commands[:, 0]) < 0.1
        )
