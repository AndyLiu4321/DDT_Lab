#!/usr/bin/env python3
"""MuJoCo Python sim2sim demo for the Mini flat velocity policy.

This script targets the 6-DoF Mini/Tita6 model used by
``DDT-Velocity-Flat-Mini-Play-v0`` and exported NP3O TorchScript policies.

The policy observation layout is the flat Mini layout:

    base_ang_vel(3), projected_gravity(3), base_velocity_command(3),
    joint_pos_rel_without_wheel(6), joint_vel(6), last_action(6)

Examples:
    python scripts/sim2sim/mini_flat_mujoco_demo.py --headless --duration 5

    python scripts/sim2sim/mini_flat_mujoco_demo.py \
        --policy logs/np3o/mini_flat/2026-07-13_10-55-51/exported/policy.pt \
        --vx 0.3 --wz 0.0 --disable-scene-obstacles
"""

from __future__ import annotations

import argparse
import importlib
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for user setup.
    raise SystemExit("Python package 'mujoco' is not installed. Run: python -m pip install mujoco") from exc

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for user setup.
    raise SystemExit("Python package 'torch' is not installed. Use the isaaclab conda environment.") from exc

from mini_jump_mujoco_demo import (
    JOINT_NAMES,
    OBS_JOINT_INDICES,
    OBS_WHEEL_JOINT_INDICES,
    REPO_ROOT,
    ControlConfig,
    disable_scene_obstacles,
    get_robot_state,
    infer_env_yaml,
    joint_addresses,
    load_control_config,
    maybe_policy_dim,
    name_id,
    pd_control,
    quat_wxyz_to_euler_xyz,
    reset_pose,
    run_policy,
)

DEFAULT_MODEL = REPO_ROOT / "ddt_ros2_control/urdfs/tita6_description/mujoco/scene.xml"
DEFAULT_POLICY = REPO_ROOT / "logs/np3o/mini_flat/2026-07-14_10-04-08/exported/policy.pt"
OBS_DIM = 27


@dataclass(frozen=True)
class Command:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


def build_observation(
    cfg: ControlConfig,
    command: Command,
    q: np.ndarray,
    dq: np.ndarray,
    omega: np.ndarray,
    projected_gravity: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs_q = q[OBS_JOINT_INDICES]
    obs_dq = dq[OBS_JOINT_INDICES]
    obs_default_dof_pos = cfg.default_dof_pos[OBS_JOINT_INDICES]
    joint_pos_rel = obs_q - obs_default_dof_pos
    joint_pos_rel[OBS_WHEEL_JOINT_INDICES] = 0.0

    obs[0:3] = omega * 0.25
    obs[3:6] = projected_gravity
    obs[6:9] = np.array([command.vx * 2.0, command.vy * 2.0, command.wz * 0.25], dtype=np.float32)
    obs[9:15] = joint_pos_rel
    obs[15:21] = obs_dq * 0.05
    obs[21:27] = last_action
    return obs


def simulate(args: argparse.Namespace) -> None:
    env_yaml = args.env_yaml
    if env_yaml is None and not args.no_auto_env_yaml:
        env_yaml = infer_env_yaml(args.policy)
    control_cfg = load_control_config(env_yaml)
    if control_cfg.source is not None:
        print(f"[mini_flat] Loaded control/action config from {control_cfg.source}")
        print(f"[mini_flat] default_dof_pos={np.array2string(control_cfg.default_dof_pos, precision=3)}")
        print(f"[mini_flat] kp={np.array2string(control_cfg.kp, precision=3)}")
        print(f"[mini_flat] kd={np.array2string(control_cfg.kd, precision=3)}")
        print(f"[mini_flat] torque_limit={np.array2string(control_cfg.torque_limit, precision=3)}")
        print(f"[mini_flat] action_scale={np.array2string(control_cfg.action_scale, precision=3)}")
        print(f"[mini_flat] action_mode={control_cfg.action_mode.tolist()}")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.opt.timestep = args.dt
    if args.disable_scene_obstacles:
        disable_scene_obstacles(model)
    data = mujoco.MjData(model)

    qpos_adrs, qvel_adrs = joint_addresses(model)
    reset_pose(control_cfg, model, data, qpos_adrs)

    device = torch.device(args.device)
    policy = torch.jit.load(str(args.policy), map_location=device)
    policy.eval()
    policy_dim = maybe_policy_dim(policy)
    if policy_dim is not None and policy_dim != OBS_DIM:
        raise RuntimeError(
            f"Expected mini_flat policy input dim {OBS_DIM}, got {policy_dim}. "
            "Use mini_jump_mujoco_demo.py for 28-dim jump policies."
        )

    q, dq, omega, projected_gravity, _ = get_robot_state(data, qpos_adrs, qvel_adrs)
    command = Command(args.vx, args.vy, args.wz)
    action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    obs = build_observation(control_cfg, command, q, dq, omega, projected_gravity, action)
    history = deque((obs.copy() for _ in range(args.history_len)), maxlen=args.history_len)

    viewer = None
    if not args.headless:
        mujoco_viewer = importlib.import_module("mujoco.viewer")
        viewer = mujoco_viewer.launch_passive(model, data)
        viewer.cam.trackbodyid = name_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -15.0

    n_steps = int(args.duration / args.dt)
    control_steps = 0
    max_base_z = float(data.qpos[2])
    min_base_z = float(data.qpos[2])
    wall_start = time.time()
    try:
        for step in range(n_steps):
            q, dq, omega, projected_gravity, quat = get_robot_state(data, qpos_adrs, qvel_adrs)
            max_base_z = max(max_base_z, float(data.qpos[2]))
            min_base_z = min(min_base_z, float(data.qpos[2]))

            if step % args.decimation == 0:
                obs = build_observation(control_cfg, command, q, dq, omega, projected_gravity, action)
                action = run_policy(policy, obs, history, device)
                history.append(obs.copy())
                action = np.clip(action, -args.clip_actions, args.clip_actions)
                control_steps += 1

            data.ctrl[:] = pd_control(control_cfg, action, q, dq)
            mujoco.mj_step(model, data)

            if viewer is not None:
                if not viewer.is_running():
                    break
                if step % args.render_every == 0:
                    viewer.sync()

            if args.print_every > 0 and step % args.print_every == 0:
                euler = quat_wxyz_to_euler_xyz(quat)
                print(
                    f"t={data.time:6.3f}s ctrl={control_steps:5d} "
                    f"cmd=({command.vx:+.2f},{command.vy:+.2f},{command.wz:+.2f}) "
                    f"base_z={data.qpos[2]:+.3f} z_range=({min_base_z:+.3f},{max_base_z:+.3f}) "
                    f"roll={euler[0]:+.2f} pitch={euler[1]:+.2f} action_norm={np.linalg.norm(action):.3f}"
                )

            if args.real_time and viewer is not None:
                delay = wall_start + data.time - time.time()
                if delay > 0.0:
                    time.sleep(delay)
    finally:
        if viewer is not None:
            viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini flat MuJoCo Python sim2sim demo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo tita6 scene.xml path.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="TorchScript mini_flat policy.pt.")
    parser.add_argument("--env-yaml", type=Path, default=None, help="Optional saved params/env.yaml.")
    parser.add_argument("--no-auto-env-yaml", action="store_true", help="Do not infer params/env.yaml from --policy.")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--decimation", type=int, default=20, help="Policy/control update interval.")
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--disable-scene-obstacles", action="store_true")
    parser.add_argument("--real-time", action="store_true", default=True)
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--clip-actions", type=float, default=100.0)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    simulate(parse_args())
