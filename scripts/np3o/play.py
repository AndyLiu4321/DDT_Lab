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
    help='Use keyboard commands during play: W/S forward/back, A/D left/right, Q/E yaw, R for jump command.',
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


class KeyboardCommandController:
    """Keyboard teleop that overwrites command observations without applying external forces."""

    def __init__(self, env: IsaacLabNP3OWrapper, lin_vel: float, ang_vel: float, source: str, hold_s: float):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
        self.command = torch.zeros(self.unwrapped.num_envs, 3, device=self.device)
        self.jump_command = torch.zeros(self.unwrapped.num_envs, 1, device=self.device)
        self._pressed_keys: set[str] = set()
        self._terminal_key_expiry: dict[str, float] = {}
        self._terminal_jump_active = False
        self._terminal_settings = None
        self._lin_vel = lin_vel
        self._ang_vel = ang_vel
        self._source = source
        self._hold_s = hold_s
        self._velocity_slice, self._velocity_scale = self._find_policy_term("velocity_commands")
        self._jump_slice, self._jump_scale = self._find_policy_term("jump")
        active_command_terms = self.unwrapped.command_manager.active_terms
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

        if self._jump_slice is None or not self._has_jump_command:
            print("[INFO] Keyboard R is captured, but this policy observation has no jump command term.")
            print("[INFO] To make R drive jumping, add a jump command observation during training/play with the same policy dimension.")
        if self._source == "terminal":
            print("[INFO] Terminal keyboard control is active. Keep this terminal focused.")
            print("[INFO] W/S forward/back, A/D left/right, Q/E yaw, R toggles jump command, Space clears commands.")
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
            if key_name in {"W", "A", "S", "D", "Q", "E", "R"}:
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
            key = ch.upper()
            if key in {"W", "A", "S", "D", "Q", "E"}:
                self._terminal_key_expiry[key] = time.monotonic() + self._hold_s
            elif key == "R":
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
        self.command[:, 0] = x * self._lin_vel
        self.command[:, 1] = y * self._lin_vel
        self.command[:, 2] = yaw * self._ang_vel
        if self._source == "terminal":
            self.jump_command[:, 0] = 1.0 if self._terminal_jump_active else 0.0
        else:
            self.jump_command[:, 0] = 1.0 if "R" in self._pressed_keys else 0.0
    def apply(self, obs: torch.Tensor) -> torch.Tensor:
        # 1. 先读取键盘，把 self.command / self.jump_command 更新到最新
        self._update_commands_from_keys()

        command_term = self.unwrapped.command_manager.get_term("base_velocity")

        # 2. 强制覆盖 command_manager 对外 command
        manager_cmd = self.unwrapped.command_manager.get_command("base_velocity")
        manager_cmd[:] = self.command

        # 3. 同时覆盖 command term 内部常见 command buffer
        #    有些 debug_vis/obs 可能读的是 term 内部变量，不是 get_command() 返回值
        if hasattr(command_term, "vel_command_b"):
            command_term.vel_command_b[:] = self.command

        if hasattr(command_term, "command"):
            command_term.command[:] = self.command

        if hasattr(command_term, "_command"):
            command_term._command[:] = self.command

        # 4. 键盘没有输入时，设置 standing_env=True
        cmd_norm = torch.norm(self.command, dim=1)

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

            if obs.dim() == 2:
                obs[:, self._velocity_slice] = scaled_command
            elif obs.dim() == 3:
                # 如果 obs 是 history 形式，覆盖所有历史帧，避免旧 command 残留
                obs[:, :, self._velocity_slice] = scaled_command[:, None, :]
            else:
                raise RuntimeError(f"Unsupported obs shape: {obs.shape}")

        # 7. 覆盖 policy observation 里的 jump command
        if self._jump_slice is not None:
            scaled_jump = self.jump_command.to(obs.device) * self._jump_scale.to(obs.device)

            if obs.dim() == 2:
                obs[:, self._jump_slice] = scaled_jump
            elif obs.dim() == 3:
                obs[:, :, self._jump_slice] = scaled_jump[:, None, :]
            else:
                raise RuntimeError(f"Unsupported obs shape: {obs.shape}")

        # 8. Debug 打印：要打印 keyboard command、manager command、obs command
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0

        self._debug_counter += 1
        if self._debug_counter % 50 == 0:
            manager_cmd_now = self.unwrapped.command_manager.get_command("base_velocity")

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
    spec = gym.spec(args_cli.task)
    runner_cfg = _resolve_runner_cfg(spec.kwargs['np3o_cfg_entry_point'])
    env_cfg_entry = spec.kwargs['env_cfg_entry_point']
    env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabNP3OWrapper(env, device=args_cli.device or 'cuda:0')

    runner = OnConstraintPolicyRunner(env, runner_cfg, log_dir=None, device=args_cli.device or 'cuda:0')

    if args_cli.checkpoint is not None:
        ckpt = args_cli.checkpoint
    else:
        log_root = os.path.abspath(os.path.join('logs', 'np3o', runner_cfg['runner']['experiment_name']))
        ckpt = get_checkpoint_path(log_root, args_cli.load_run, args_cli.load_checkpoint)
    print(f'[INFO] loading checkpoint: {ckpt}')
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
                obs, _, _, _, _, _ = env.step(actions)
                if keyboard_controller is not None:
                    obs = keyboard_controller.apply(obs)
    finally:
        if keyboard_controller is not None:
            keyboard_controller.close()
    env.env.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
