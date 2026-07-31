# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play / evaluate a checkpoint trained with NP3O."""

import argparse
import atexit
import importlib
import os
import select
import sys
import termios
import time
import tty

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='Play a checkpoint trained with NP3O.')
parser.add_argument('--task', type=str, required=True, help='Gym task ID.')
parser.add_argument('--num_envs', type=int, default=50, help='Number of envs for playback.')
parser.add_argument('--checkpoint', type=str, default=None, help='Absolute checkpoint path (overrides auto-resolve).')
parser.add_argument('--load_run', type=str, default='.*', help='Run dir regex when --checkpoint is omitted.')
parser.add_argument('--load_checkpoint', type=str, default=r'model_.*\.pt', help='Checkpoint filename regex.')
parser.add_argument(
    '--export_policy', action='store_true',
    help='Export the loaded policy as TorchScript + ONNX next to the checkpoint and exit.',
)
parser.add_argument(
    '--export_dir', type=str, default=None,
    help='Override export directory (defaults to <checkpoint_dir>/exported).',
)
parser.add_argument(
    '--keyboard', action='store_true',
    help='Use keyboard commands during play. Command-gated Tita uses 6-D commands; legacy jump tasks keep R for jump.',
)
parser.add_argument(
    '--allow_mismatched_checkpoint', action='store_true',
    help='Allow partial checkpoint loading when observation dimensions do not match. Unsafe for normal play.',
)
parser.add_argument(
    '--keyboard_source',
    type=str,
    default='terminal',
    choices=('terminal', 'app'),
    help='Read keyboard input from the terminal or from the Isaac Sim app window.',
)
parser.add_argument('--keyboard_lin_vel', type=float, default=0.6, help='Keyboard linear velocity command magnitude.')
parser.add_argument('--keyboard_ang_vel', type=float, default=0.8, help='Keyboard yaw velocity command magnitude.')
parser.add_argument('--keyboard_hold_s', type=float, default=0.25, help='Terminal key hold time between key repeats.')
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import ddt_lab.tasks  # noqa: F401  -- registers tasks
import gymnasium as gym
import torch
from isaaclab_tasks.utils import get_checkpoint_path

from ddt_lab.algorithms.np3o import IsaacLabNP3OWrapper, OnConstraintPolicyRunner


_KNOWN_MINI_TASK_OBS_DIMS = {
    "DDT-Velocity-Flat-Mini-v0": (27, 44),
    "DDT-Velocity-Flat-Mini-Play-v0": (27, 44),
    "DDT-Recovery-Flat-Mini-v0": (27, 231),
    "DDT-Recovery-Flat-Mini-Play-v0": (27, 231),
    "DDT-jump-Flat-Mini-v0": (28, 49),
    "DDT-jump-Flat-Mini-Play-v0": (28, 49),
}


def _checkpoint_obs_dims(checkpoint_path: str) -> tuple[int, int] | None:
    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except RuntimeError:
        return None
    if not isinstance(loaded, dict) or "model_state_dict" not in loaded:
        return None
    state_dict = loaded["model_state_dict"]
    actor_key = "actor_teacher_backbone.obs_normalizer._mean"
    critic_key = "critic_obs_normalizer._mean"
    if actor_key not in state_dict or critic_key not in state_dict:
        return None
    return int(state_dict[actor_key].shape[-1]), int(state_dict[critic_key].shape[-1])


def _auto_select_task_for_checkpoint(task: str, checkpoint_dims: tuple[int, int] | None) -> str:
    if checkpoint_dims is None or "Mini" not in task:
        return task
    expected_dims = _KNOWN_MINI_TASK_OBS_DIMS.get(task)
    if expected_dims == checkpoint_dims:
        return task
    if expected_dims is None:
        print(
            "[WARN] Cannot infer the requested Mini task family from observation dimensions alone; "
            f"keeping {task}."
        )
        return task

    prefer_play = task.endswith("-Play-v0")
    candidate_tasks = [
        "DDT-Velocity-Flat-Mini-Play-v0" if prefer_play else "DDT-Velocity-Flat-Mini-v0",
        "DDT-Recovery-Flat-Mini-Play-v0" if prefer_play else "DDT-Recovery-Flat-Mini-v0",
        "DDT-jump-Flat-Mini-Play-v0" if prefer_play else "DDT-jump-Flat-Mini-v0",
    ]
    matches = [candidate for candidate in candidate_tasks if _KNOWN_MINI_TASK_OBS_DIMS[candidate] == checkpoint_dims]
    if len(matches) == 1:
        candidate = matches[0]
        print(
            "[INFO] checkpoint observation dims "
            f"policy={checkpoint_dims[0]}, critic={checkpoint_dims[1]} match {candidate}; "
            f"switching task from {task} to {candidate}."
        )
        return candidate
    if matches:
        print(
            "[WARN] checkpoint observation dims "
            f"policy={checkpoint_dims[0]}, critic={checkpoint_dims[1]} match multiple Mini tasks "
            f"({', '.join(matches)}); keeping requested task {task}."
        )
    else:
        print(
            "[WARN] checkpoint observation dims "
            f"policy={checkpoint_dims[0]}, critic={checkpoint_dims[1]} do not match {task} "
            f"(policy={expected_dims[0]}, critic={expected_dims[1]}), and no known Mini task matched."
        )
    return task


def _runner_obs_dims(runner: OnConstraintPolicyRunner) -> tuple[int, int]:
    actor_critic = runner.alg.actor_critic
    return int(actor_critic.num_prop), int(actor_critic.num_critic_obs)


def _validate_checkpoint_dims(
    checkpoint_path: str,
    checkpoint_dims: tuple[int, int] | None,
    runner: OnConstraintPolicyRunner,
):
    if checkpoint_dims is None:
        return
    model_dims = _runner_obs_dims(runner)
    if checkpoint_dims == model_dims:
        return
    msg = (
        "Checkpoint observation dimensions do not match the current play task.\n"
        f"  checkpoint: {checkpoint_path}\n"
        f"  checkpoint dims: policy={checkpoint_dims[0]}, critic={checkpoint_dims[1]}\n"
        f"  task: {args_cli.task}\n"
        f"  model dims: policy={model_dims[0]}, critic={model_dims[1]}\n"
        "Use the matching Play task for this checkpoint, or pass "
        "--allow_mismatched_checkpoint if you intentionally want partial/random-initialized loading."
    )
    if args_cli.allow_mismatched_checkpoint:
        print(f"[WARN] {msg}")
        return
    raise RuntimeError(msg)


class KeyboardCommandController:
    """Keyboard teleop that overwrites command observations without applying external forces."""

    def __init__(self, env: IsaacLabNP3OWrapper, lin_vel: float, ang_vel: float, source: str, hold_s: float):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
        active_command_terms = self.unwrapped.command_manager.active_terms
        self._wheel_command_name = "wheel_commands" if "wheel_commands" in active_command_terms else None
        self._velocity_command_name = self._wheel_command_name or ("base_velocity" if "base_velocity" in active_command_terms else None)
        self._is_wheel_command = self._wheel_command_name is not None
        self.command = torch.zeros(self.unwrapped.num_envs, 6 if self._is_wheel_command else 3, device=self.device)
        self.jump_command = torch.zeros(self.unwrapped.num_envs, 1, device=self.device)
        self._velocity_history: torch.Tensor | None = None
        self._jump_history: torch.Tensor | None = None
        self._pressed_keys: set[str] = set()
        self._terminal_key_expiry: dict[str, float] = {}
        self._terminal_jump_active = False
        self._leg_select: str | None = None
        self._tsk_cmd = 0.0
        self._reset_requested = False
        self._terminal_settings = None
        self._lin_vel = lin_vel
        self._ang_vel = ang_vel
        self._source = source
        self._hold_s = hold_s
        self._velocity_slice, self._velocity_scale = self._find_policy_term(
            "wheel_commands" if self._is_wheel_command else "velocity_commands"
        )
        self._jump_slice, self._jump_scale = self._find_policy_term("jump")
        if "jump_cmd" in active_command_terms:
            self._jump_command_name = "jump_cmd"
        elif "jump" in active_command_terms:
            self._jump_command_name = "jump"
        else:
            self._jump_command_name = None
        self._has_jump_command = self._jump_command_name is not None
        if self._has_jump_command:
            jump_term = self.unwrapped.command_manager.get_term(self._jump_command_name)
            if hasattr(jump_term.cfg, "manual_mode"):
                jump_term.cfg.manual_mode = True
            print("[INFO] Jump command manual mode is active; use R to toggle jump.")

        if self._source == "app":
            import carb
            import omni.appwindow

            self._carb = carb
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        else:
            self._setup_terminal_keyboard()

        if (self._jump_slice is None or not self._has_jump_command) and not self._is_wheel_command:
            print("[INFO] Keyboard R is captured, but this policy observation has no jump command term.")
            print("[INFO] To make R drive jumping, add a jump command observation during training/play with the same policy dimension.")
        if self._source == "terminal":
            print("[INFO] Terminal keyboard control is active. Keep this terminal focused.")
            if self._is_wheel_command:
                print("[INFO] W/S vx, A/D vy, Q/E wz, Space up, C/Ctrl down, 1/2 single leg, arrows tsk, Backspace tsk=0, Shift slow, R reset.")
            else:
                print("[INFO] W/S forward/back, A/D left/right, Q/E yaw, R toggles jump command, Space clears commands.")
        else:
            if self._is_wheel_command:
                print("[INFO] Isaac app keyboard control: W/S/A/D/Q/E velocity, Space/Ctrl/C height, arrows tsk, Backspace tsk=0, R reset.")
            else:
                print("[INFO] Isaac app keyboard control: W/S forward/back, A/D left/right, Q/E yaw, hold R for jump command.")

    def _setup_terminal_keyboard(self):
        if not sys.stdin.isatty():
            print("[WARN] stdin is not a TTY; terminal keyboard control is disabled.")
            return
        self._terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        atexit.register(self.close)

    def close(self):
        if self._terminal_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._terminal_settings)
            self._terminal_settings = None

    def _find_policy_term(self, name_part: str):
        obs_manager = self.unwrapped.observation_manager
        term_names = obs_manager.active_terms.get("policy", [])
        term_dims = obs_manager.group_obs_term_dim.get("policy", [])
        term_cfgs = obs_manager._group_obs_term_cfgs.get("policy", [])
        start = 0
        for term_name, term_dim, term_cfg in zip(term_names, term_dims, term_cfgs):
            if isinstance(term_dim, int):
                width = term_dim
            else:
                width = int(term_dim[-1]) if len(term_dim) > 0 else 0
            if name_part in term_name:
                if width <= 0:
                    print(f"[WARN] Skipping empty policy observation term '{term_name}' with dim {term_dim}.")
                    return None, None
                scale = term_cfg.scale
                if scale is None:
                    scale_tensor = torch.ones(width, device=self.device)
                elif isinstance(scale, torch.Tensor):
                    scale_tensor = scale.to(device=self.device, dtype=torch.float32).reshape(-1)
                elif isinstance(scale, (tuple, list)):
                    scale_tensor = torch.tensor(scale, device=self.device, dtype=torch.float32).reshape(-1)
                else:
                    scale_tensor = torch.full((width,), float(scale), device=self.device)
                if scale_tensor.numel() == 1 and width != 1:
                    scale_tensor = scale_tensor.repeat(width)
                if scale_tensor.numel() != width:
                    scale_tensor = scale_tensor[:width]
                return slice(start, start + width), scale_tensor
            start += width
        return None, None

    def _on_keyboard_event(self, event):
        key_name = event.input.name.upper()
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            if self._is_wheel_command and key_name == "R":
                self._reset_requested = True
                return
            if key_name in {
                "W", "A", "S", "D", "Q", "E", "R", "SPACE", "C", "LEFT_CONTROL", "RIGHT_CONTROL",
                "LEFT_SHIFT", "RIGHT_SHIFT", "SHIFT", "LEFT", "RIGHT", "LEFT_ARROW", "RIGHT_ARROW",
                "BACKSPACE", "1", "2", "KEY_1", "KEY_2",
            }:
                self._pressed_keys.add(key_name)
        elif event.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed_keys.discard(key_name)

    def _poll_terminal_events(self):
        if self._source != "terminal" or self._terminal_settings is None:
            return
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                break
            ch = sys.stdin.read(1)
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x1b":
                seq = ch + sys.stdin.read(2)
                if seq == "\x1b[D":
                    self._terminal_key_expiry["LEFT"] = time.monotonic() + self._hold_s
                elif seq == "\x1b[C":
                    self._terminal_key_expiry["RIGHT"] = time.monotonic() + self._hold_s
                continue
            key = ch.upper()
            if key in {"W", "A", "S", "D", "Q", "E"}:
                self._terminal_key_expiry[key] = time.monotonic() + self._hold_s
            elif key == "\x7f":
                self._tsk_cmd = 0.0
                print("\r[INFO] tsk command centered", end="", flush=True)
            elif key == "\x1b":
                self._terminal_key_expiry["SHIFT"] = time.monotonic() + self._hold_s
            elif key == " " and self._is_wheel_command:
                self._terminal_key_expiry["SPACE"] = time.monotonic() + self._hold_s
            elif key in {"C", "\x0c"} and self._is_wheel_command:
                self._terminal_key_expiry["C"] = time.monotonic() + self._hold_s
            elif key == "1" and self._is_wheel_command:
                self._leg_select = "left"
                print("\r[INFO] leg command target: left ", end="", flush=True)
            elif key == "2" and self._is_wheel_command:
                self._leg_select = "right"
                print("\r[INFO] leg command target: right", end="", flush=True)
            elif key == "R":
                if self._is_wheel_command:
                    self._reset_requested = True
                    print("\r[INFO] reset requested", end="", flush=True)
                else:
                    self._terminal_jump_active = not self._terminal_jump_active
                    print(f"\r[INFO] jump command {'ON ' if self._terminal_jump_active else 'OFF'}", end="", flush=True)
            elif ch == " ":
                self._terminal_key_expiry.clear()
                self._terminal_jump_active = False
                self.command.zero_()
                self.jump_command.zero_()
                print("\r[INFO] commands cleared", end="", flush=True)

    def _update_commands_from_keys(self):
        self._poll_terminal_events()
        if self._source == "terminal":
            now = time.monotonic()
            active = {key for key, expiry in self._terminal_key_expiry.items() if expiry >= now}
            self._terminal_key_expiry = {key: expiry for key, expiry in self._terminal_key_expiry.items() if expiry >= now}
        else:
            active = self._pressed_keys
        x = (1.0 if "W" in active else 0.0) - (1.0 if "S" in active else 0.0)
        y = (1.0 if "A" in active else 0.0) - (1.0 if "D" in active else 0.0)
        yaw = (1.0 if "Q" in active else 0.0) - (1.0 if "E" in active else 0.0)
        slow = bool({"LEFT_SHIFT", "RIGHT_SHIFT", "SHIFT"} & set(active))
        lin_scale = 0.35 if slow else 1.0
        ang_scale = 0.35 if slow else 1.0
        self.command[:, 0] = x * self._lin_vel * lin_scale
        self.command[:, 1] = y * self._lin_vel * lin_scale
        self.command[:, 2] = yaw * self._ang_vel * ang_scale
        if self._is_wheel_command:
            up = "SPACE" in active or " " in active
            down = bool({"C", "LEFT_CONTROL", "RIGHT_CONTROL"} & set(active))
            if up or down:
                target = 0.9 if up else 0.2
                app_left = bool({"1", "KEY_1"} & set(active))
                app_right = bool({"2", "KEY_2"} & set(active))
                leg_select = "left" if app_left else "right" if app_right else self._leg_select
                if leg_select == "left":
                    self.command[:, 3] = target
                elif leg_select == "right":
                    self.command[:, 4] = target
                else:
                    self.command[:, 3] = target
                    self.command[:, 4] = target
            if "LEFT" in active or "LEFT_ARROW" in active:
                self._tsk_cmd = -0.3
            elif "RIGHT" in active or "RIGHT_ARROW" in active:
                self._tsk_cmd = 0.3
            if "BACKSPACE" in active:
                self._tsk_cmd = 0.0
            self.command[:, 5] = self._tsk_cmd
        if self._source == "terminal":
            self.jump_command[:, 0] = 1.0 if self._terminal_jump_active else 0.0
        else:
            self.jump_command[:, 0] = 1.0 if "R" in self._pressed_keys else 0.0

    def _overwrite_policy_term(
        self,
        obs: torch.Tensor,
        term_slice: slice | None,
        scaled_value: torch.Tensor,
        history_attr: str,
        advance_history: bool,
        reset_mask: torch.Tensor | None,
    ) -> None:
        if term_slice is None:
            return
        if obs.dim() == 2:
            obs[:, term_slice] = scaled_value
            return
        if obs.dim() != 3:
            raise RuntimeError(f"Unsupported obs shape: {obs.shape}")

        history = getattr(self, history_attr)
        expected_shape = (obs.shape[0], obs.shape[1], scaled_value.shape[-1])
        if history is None or tuple(history.shape) != expected_shape:
            history = scaled_value[:, None, :].expand(expected_shape).clone()
        elif advance_history:
            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1, :] = scaled_value
        else:
            history[:, -1, :] = scaled_value

        if reset_mask is not None and torch.any(reset_mask):
            history[reset_mask] = scaled_value[reset_mask, None, :]

        obs[:, :, term_slice] = history
        setattr(self, history_attr, history)

    def apply(
        self,
        obs: torch.Tensor,
        *,
        advance_history: bool = False,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. 先读取键盘，把 self.command / self.jump_command 更新到最新
        self._update_commands_from_keys()

        if self._reset_requested:
            obs = self.env.reset()
            self._reset_requested = False
            reset_mask = torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)

        if reset_mask is not None:
            reset_mask = reset_mask.to(device=obs.device, dtype=torch.bool).reshape(-1)

        if self._velocity_command_name is None:
            return obs

        command_term = self.unwrapped.command_manager.get_term(self._velocity_command_name)

        # 2. 强制覆盖 command_manager 对外 command
        manager_cmd = self.unwrapped.command_manager.get_command(self._velocity_command_name)
        manager_cmd[:] = self.command

        # 3. 同时覆盖 command term 内部常见 command buffer
        #    有些 debug_vis/obs 可能读的是 term 内部变量，不是 get_command() 返回值
        if hasattr(command_term, "vel_command_b"):
            command_term.vel_command_b[:] = self.command[:, : command_term.vel_command_b.shape[-1]]

        if hasattr(command_term, "command_buf"):
            command_term.command_buf[:] = self.command

        if hasattr(command_term, "command"):
            try:
                command_term.command[:] = self.command
            except AttributeError:
                pass

        if hasattr(command_term, "_command"):
            command_term._command[:] = self.command

        # 4. 键盘没有输入时，设置 standing_env=True
        cmd_norm = torch.norm(self.command[:, :3], dim=1)

        if hasattr(command_term, "is_heading_env"):
            command_term.is_heading_env[:] = False

        if hasattr(command_term, "is_standing_env"):
            command_term.is_standing_env[:] = cmd_norm < 1e-4

        # 5. 覆盖 jump command
        if self._has_jump_command:
            jump_term = self.unwrapped.command_manager.get_term(self._jump_command_name)
            jump_term.jump_cmd[:] = self.jump_command

            if hasattr(jump_term, "command"):
                jump_term.command[:] = self.jump_command

            if hasattr(jump_term, "_command"):
                jump_term._command[:] = self.jump_command

        # 6. 覆盖 policy observation 里的 velocity command
        if self._velocity_slice is not None:
            scaled_command = self.command.to(obs.device) * self._velocity_scale.to(obs.device)
            self._overwrite_policy_term(
                obs,
                self._velocity_slice,
                scaled_command,
                "_velocity_history",
                advance_history,
                reset_mask,
            )

        # 7. 覆盖 policy observation 里的 jump command
        if self._jump_slice is not None:
            scaled_jump = self.jump_command.to(obs.device) * self._jump_scale.to(obs.device)
            self._overwrite_policy_term(
                obs,
                self._jump_slice,
                scaled_jump,
                "_jump_history",
                advance_history,
                reset_mask,
            )

        # 8. Debug 打印：要打印 keyboard command、manager command、obs command
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0

        self._debug_counter += 1
        if self._debug_counter % 50 == 0:
            manager_cmd_now = self.unwrapped.command_manager.get_command(self._velocity_command_name)

            if self._is_wheel_command:
                msg = (
                    "\r[CMD] "
                    "vx={:.2f}, vy={:.2f}, wz={:.2f}, left={:.2f}, right={:.2f}, tsk={:.2f} | "
                    "manager vx={:.2f}, vy={:.2f}, wz={:.2f}, left={:.2f}, right={:.2f}, tsk={:.2f}"
                ).format(
                    self.command[0, 0].item(),
                    self.command[0, 1].item(),
                    self.command[0, 2].item(),
                    self.command[0, 3].item(),
                    self.command[0, 4].item(),
                    self.command[0, 5].item(),
                    manager_cmd_now[0, 0].item(),
                    manager_cmd_now[0, 1].item(),
                    manager_cmd_now[0, 2].item(),
                    manager_cmd_now[0, 3].item(),
                    manager_cmd_now[0, 4].item(),
                    manager_cmd_now[0, 5].item(),
                )
            else:
                msg = (
                    "\r[CMD] "
                    "keyboard vx={:.2f}, vy={:.2f}, wz={:.2f}, jump={:.1f} | "
                    "manager vx={:.2f}, vy={:.2f}, wz={:.2f} | "
                    "standing={}"
                ).format(
                    self.command[0, 0].item(),
                    self.command[0, 1].item(),
                    self.command[0, 2].item(),
                    self.jump_command[0, 0].item(),
                    manager_cmd_now[0, 0].item(),
                    manager_cmd_now[0, 1].item(),
                    manager_cmd_now[0, 2].item(),
                    bool(cmd_norm[0].item() < 1e-4),
                )

            if self._velocity_slice is not None:
                if obs.dim() == 2:
                    obs_cmd = obs[0, self._velocity_slice].detach().cpu().numpy()
                elif obs.dim() == 3:
                    obs_cmd = obs[0, -1, self._velocity_slice].detach().cpu().numpy()
                else:
                    obs_cmd = None

                msg += f" | obs_cmd={obs_cmd}"

            print(msg, end="", flush=True)

        return obs

def _resolve_runner_cfg(entry_point: str) -> dict:
    module_name, attr = entry_point.split(':')
    obj = getattr(importlib.import_module(module_name), attr)
    cfg = obj() if callable(obj) else obj
    if not isinstance(cfg, dict):
        raise TypeError(f"NP3O cfg entry point '{entry_point}' must return a dict")
    return cfg


def main():
    requested_task = args_cli.task
    spec = gym.spec(requested_task)
    runner_cfg = _resolve_runner_cfg(spec.kwargs['np3o_cfg_entry_point'])
    if args_cli.checkpoint is not None:
        ckpt = args_cli.checkpoint
    else:
        log_root = os.path.abspath(os.path.join('logs', 'np3o', runner_cfg['runner']['experiment_name']))
        ckpt = get_checkpoint_path(log_root, args_cli.load_run, args_cli.load_checkpoint)

    checkpoint_dims = _checkpoint_obs_dims(ckpt)
    args_cli.task = _auto_select_task_for_checkpoint(requested_task, checkpoint_dims)

    spec = gym.spec(args_cli.task)
    runner_cfg = _resolve_runner_cfg(spec.kwargs['np3o_cfg_entry_point'])
    env_cfg_entry = spec.kwargs['env_cfg_entry_point']
    env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabNP3OWrapper(env, device=args_cli.device or 'cuda:0')

    runner = OnConstraintPolicyRunner(env, runner_cfg, log_dir=None, device=args_cli.device or 'cuda:0')
    print(f'[INFO] loading checkpoint: {ckpt}')
    _validate_checkpoint_dims(ckpt, checkpoint_dims, runner)
    runner.load(ckpt, load_optimizer=False)

    # Always export the JIT/ONNX policy next to the checkpoint (matches the
    # pre-NP3O scripts/rsl_rl/play.py behavior). Skip on --export_policy off
    # would be a one-line guard; we keep it on by default since it's cheap.
    export_dir = args_cli.export_dir or os.path.join(os.path.dirname(ckpt), 'exported')
    runner.alg.actor_critic.save_torch_jit_policy(export_dir, args_cli.device or 'cuda:0')
    if args_cli.export_policy:
        # --export_policy explicitly requested: skip rollout, just export.
        env.env.close()
        return

    policy = runner.get_inference_policy(args_cli.device or 'cuda:0')
    obs = env.get_observations()
    keyboard_controller = KeyboardCommandController(
        env,
        lin_vel=args_cli.keyboard_lin_vel,
        ang_vel=args_cli.keyboard_ang_vel,
        source=args_cli.keyboard_source,
        hold_s=args_cli.keyboard_hold_s,
    ) if args_cli.keyboard else None
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if keyboard_controller is not None:
                    obs = keyboard_controller.apply(obs)
                actions = policy(obs)
                obs, _, _, _, dones, _ = env.step(actions)
                if keyboard_controller is not None:
                    obs = keyboard_controller.apply(obs, advance_history=True, reset_mask=dones)
    finally:
        if keyboard_controller is not None:
            keyboard_controller.close()
    env.env.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
