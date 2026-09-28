#!/usr/bin/env python3
"""MuJoCo Python sim2sim demo for Tita.

Examples:
    python scripts/sim2sim/tita_mujoco_demo.py --headless --duration 2

    python scripts/sim2sim/tita_mujoco_demo.py \
        --policy logs/np3o/tita_command_gated_flat/2026-07-09_10-13-00/exported/policy.pt \
        --profile command_gated --vx 1 \
        --duration 50
    python scripts/sim2sim/tita_mujoco_demo.py \
        --policy logs/np3o/tita_rough/2026-06-17_15-55-43/exported/policy.pt \
        --profile velocity --vx 1

    # R toggles jump_cmd between 0 and 1.
    python scripts/sim2sim/tita_mujoco_demo.py --profile jump --vx 0
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import re
import select
import sys
import termios
import time
import tty
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for user setup.
    raise SystemExit(
        "Python package 'mujoco' is not installed in this environment. "
        "Run: python -m pip install mujoco"
    ) from exc

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for user setup.
    raise SystemExit(
        "Python package 'torch' is not installed in this environment. "
        "Use the isaaclab conda environment before running this script."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "ddt_ros2_control/urdfs/tita_description/mujoco/scene.xml"
DEFAULT_JUMP_POLICY = (
    REPO_ROOT / "logs/np3o/tita_jump/2026-07-16_17-31-02/exported/policy.pt"
)

JOINT_NAMES = (
    "joint_left_leg_1",
    "joint_left_leg_2",
    "joint_left_leg_3",
    "joint_left_leg_4",
    "joint_right_leg_1",
    "joint_right_leg_2",
    "joint_right_leg_3",
    "joint_right_leg_4",
)
LEG_JOINT_INDICES = np.array([0, 1, 2, 4, 5, 6], dtype=np.int64)
WHEEL_JOINT_INDICES = np.array([3, 7], dtype=np.int64)
# The velocity profile observes IsaacLab's articulation order, grouped by
# joint number: [L1, R1, L2, R2, L3, R3, L4, R4].
VELOCITY_OBS_JOINT_INDICES = np.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int64)
VELOCITY_OBS_WHEEL_JOINT_INDICES = np.array([6, 7], dtype=np.int64)

DEFAULT_DOF_POS = np.array([0.1, 0.8, -1.5, 0.0, 0.1, 0.8, -1.5, 0.0], dtype=np.float64)
KP = np.array([60.0, 60.0, 60.0, 0.0, 60.0, 60.0, 60.0, 0.0], dtype=np.float64)
KD = np.array([1.5, 1.5, 1.5, 0.5, 1.5, 1.5, 1.5, 0.5], dtype=np.float64)
TORQUE_LIMIT = np.array([60.0, 60.0, 60.0, 12.0, 60.0, 60.0, 60.0, 12.0], dtype=np.float64)

LEG_ACTION_SCALE = 0.25
WHEEL_ACTION_SCALE = 5.0
DEFAULT_ACTION_SCALE = np.array(
    [
        LEG_ACTION_SCALE,
        LEG_ACTION_SCALE,
        LEG_ACTION_SCALE,
        WHEEL_ACTION_SCALE,
        LEG_ACTION_SCALE,
        LEG_ACTION_SCALE,
        LEG_ACTION_SCALE,
        WHEEL_ACTION_SCALE,
    ],
    dtype=np.float64,
)
DEFAULT_ACTION_MODE = np.array(["pos", "pos", "pos", "vel", "pos", "pos", "pos", "vel"], dtype=object)


@dataclass(frozen=True)
class Command:
    vx: float = 0.4
    vy: float = 0.0
    wz: float = 0.0
    left_leg_length: float = 0.5
    right_leg_length: float = 0.5
    tsk: float = 0.0
    jump_cmd: float = 0.0


@dataclass
class ControlConfig:
    default_dof_pos: np.ndarray = field(default_factory=lambda: DEFAULT_DOF_POS.copy())
    root_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.40], dtype=np.float64))
    root_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    kp: np.ndarray = field(default_factory=lambda: KP.copy())
    kd: np.ndarray = field(default_factory=lambda: KD.copy())
    torque_limit: np.ndarray = field(default_factory=lambda: TORQUE_LIMIT.copy())
    action_scale: np.ndarray = field(default_factory=lambda: DEFAULT_ACTION_SCALE.copy())
    action_mode: np.ndarray = field(default_factory=lambda: DEFAULT_ACTION_MODE.copy())
    source: Path | None = None


def infer_env_yaml(policy_path: Path | None) -> Path | None:
    if policy_path is None:
        return None
    candidate = policy_path.resolve().parents[1] / "params" / "env.yaml"
    return candidate if candidate.exists() else None


def joint_regex_matches(pattern: str, joint_name: str) -> bool:
    try:
        return re.fullmatch(pattern, joint_name) is not None or re.match(pattern, joint_name) is not None
    except re.error:
        return pattern == joint_name


def load_control_config(env_yaml: Path | None) -> ControlConfig:
    cfg = ControlConfig(source=env_yaml)
    if env_yaml is None:
        return cfg

    with env_yaml.open("r", encoding="utf-8") as f:
        env_cfg = yaml.unsafe_load(f)

    robot_cfg = env_cfg.get("scene", {}).get("robot", {})
    init_state = robot_cfg.get("init_state", {})
    joint_pos = init_state.get("joint_pos", {})
    root_pos = init_state.get("pos")
    root_quat = init_state.get("rot")
    if root_pos is not None:
        cfg.root_pos = np.asarray(root_pos, dtype=np.float64)
    if root_quat is not None:
        cfg.root_quat = np.asarray(root_quat, dtype=np.float64)
    for i, name in enumerate(JOINT_NAMES):
        if name in joint_pos:
            cfg.default_dof_pos[i] = float(joint_pos[name])

    cfg.kp[:] = 0.0
    cfg.kd[:] = 0.0
    cfg.torque_limit[:] = np.inf
    for actuator in robot_cfg.get("actuators", {}).values():
        patterns = actuator.get("joint_names_expr", [])
        stiffness = actuator.get("stiffness")
        damping = actuator.get("damping")
        effort_limit = actuator.get("effort_limit_sim", None)
        if effort_limit is None:
            effort_limit = actuator.get("effort_limit", None)
        if effort_limit is None:
            effort_limit = actuator.get("saturation_effort", None)
        for i, name in enumerate(JOINT_NAMES):
            if any(joint_regex_matches(pattern, name) for pattern in patterns):
                if stiffness is not None:
                    cfg.kp[i] = float(stiffness)
                if damping is not None:
                    cfg.kd[i] = float(damping)
                if effort_limit is not None:
                    cfg.torque_limit[i] = float(effort_limit)

    cfg.action_scale[:] = 0.0
    for action in env_cfg.get("actions", {}).values():
        joint_names = action.get("joint_names", [])
        scale = action.get("scale", None)
        class_type = str(action.get("class_type", ""))
        if scale is None:
            continue
        mode = "vel" if "JointVelocityAction" in class_type else "pos"
        for joint_name in joint_names:
            for i, name in enumerate(JOINT_NAMES):
                if joint_regex_matches(joint_name, name):
                    cfg.action_scale[i] = float(scale)
                    cfg.action_mode[i] = mode

    # Keep a conservative fallback if an older YAML omits action entries.
    missing_scale = cfg.action_scale == 0.0
    cfg.action_scale[missing_scale] = DEFAULT_ACTION_SCALE[missing_scale]
    cfg.action_mode[missing_scale] = DEFAULT_ACTION_MODE[missing_scale]
    return cfg


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Return rotation matrix from body frame to world frame."""
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(float(pitch_arg))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def name_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"MuJoCo model is missing {name!r}")
    return obj_id


def joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos_adrs = []
    qvel_adrs = []
    for name in JOINT_NAMES:
        joint_id = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adrs.append(model.jnt_qposadr[joint_id])
        qvel_adrs.append(model.jnt_dofadr[joint_id])
    return np.asarray(qpos_adrs, dtype=np.int64), np.asarray(qvel_adrs, dtype=np.int64)


def sensor_vec(data: mujoco.MjData, name: str) -> np.ndarray:
    return np.asarray(data.sensor(name).data, dtype=np.float64).copy()


def get_robot_state(
    model: mujoco.MjModel, data: mujoco.MjData, qpos_adrs: np.ndarray, qvel_adrs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    quat = sensor_vec(data, "trunk_quat")
    if quat[0] < 0.0:
        quat = -quat
    rot = quat_wxyz_to_rotmat(quat)
    projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

    q = data.qpos[qpos_adrs].astype(np.float64).copy()
    dq = data.qvel[qvel_adrs].astype(np.float64).copy()
    omega = sensor_vec(data, "trunk_gyro")

    return q, dq, omega, projected_gravity, quat


def build_observation(cfg: ControlConfig, profile: str, command: Command, q: np.ndarray, dq: np.ndarray, omega: np.ndarray,
                      projected_gravity: np.ndarray, last_action: np.ndarray) -> np.ndarray:
    if profile == "command_gated":
        obs = np.zeros(34, dtype=np.float32)
        obs[0:3] = omega * 0.25
        obs[3:6] = projected_gravity
        obs[6:12] = (q[LEG_JOINT_INDICES] - cfg.default_dof_pos[LEG_JOINT_INDICES])
        obs[12:20] = dq * 0.05
        obs[20:28] = last_action
        obs[28:34] = np.array(
            [
                command.vx * 2.0,
                command.vy * 2.0,
                command.wz * 0.25,
                command.left_leg_length,
                command.right_leg_length,
                command.tsk,
            ],
            dtype=np.float32,
        )
        return obs

    if profile in ("velocity", "jump"):
        obs = np.zeros(34 if profile == "jump" else 33, dtype=np.float32)
        obs_q = q[VELOCITY_OBS_JOINT_INDICES]
        obs_dq = dq[VELOCITY_OBS_JOINT_INDICES]
        obs_default_dof_pos = cfg.default_dof_pos[VELOCITY_OBS_JOINT_INDICES]
        joint_pos_rel = obs_q - obs_default_dof_pos
        joint_pos_rel[VELOCITY_OBS_WHEEL_JOINT_INDICES] = 0.0
        obs[0:3] = omega * 0.25
        obs[3:6] = projected_gravity
        obs[6:9] = np.array([command.vx * 2.0, command.vy * 2.0, command.wz * 0.25], dtype=np.float32)
        obs[9:17] = joint_pos_rel
        obs[17:25] = obs_dq * 0.05
        obs[25:33] = last_action
        if profile == "jump":
            obs[33] = command.jump_cmd
        return obs

    raise ValueError(f"Unknown profile: {profile}")


def pd_control(cfg: ControlConfig, action: np.ndarray, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    target_q = cfg.default_dof_pos.copy()
    target_dq = np.zeros_like(cfg.default_dof_pos)

    pos_mask = cfg.action_mode == "pos"
    vel_mask = cfg.action_mode == "vel"
    target_q[pos_mask] += action[pos_mask] * cfg.action_scale[pos_mask]
    target_dq[vel_mask] = action[vel_mask] * cfg.action_scale[vel_mask]

    tau = (target_q - q) * cfg.kp + (target_dq - dq) * cfg.kd
    return np.clip(tau, -cfg.torque_limit, cfg.torque_limit)


def run_policy(policy: torch.jit.ScriptModule, current: np.ndarray, history: deque[np.ndarray],
               device: torch.device) -> np.ndarray:
    current_t = torch.as_tensor(current, dtype=torch.float32, device=device).unsqueeze(0)
    history_t = torch.as_tensor(np.stack(history), dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        try:
            action = policy(current_t, history_t)
        except RuntimeError as exc:
            flat_history_t = history_t.reshape(1, -1)
            try:
                action = policy(flat_history_t)
            except RuntimeError:
                raise RuntimeError(
                    f"Policy input failed. Current shape={tuple(current_t.shape)}, "
                    f"history shape={tuple(history_t.shape)}, flat history shape={tuple(flat_history_t.shape)}. "
                    "Check --profile and --history-len."
                ) from exc

    return action.detach().cpu().numpy()[0].astype(np.float64)


def reset_pose(
    cfg: ControlConfig, model: mujoco.MjModel, data: mujoco.MjData, qpos_adrs: np.ndarray, qvel_adrs: np.ndarray
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = cfg.root_pos
    data.qpos[3:7] = cfg.root_quat
    data.qpos[qpos_adrs] = cfg.default_dof_pos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def disable_scene_obstacles(model: mujoco.MjModel) -> None:
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        is_world_box = model.geom_bodyid[geom_id] == 0 and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
        if name != "floor" and is_world_box:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            model.geom_rgba[geom_id, 3] = 0.15


def simulate(args: argparse.Namespace) -> None:
    if args.profile == "jump" and args.policy is None:
        args.policy = DEFAULT_JUMP_POLICY
        print(f"[sim2sim] Using jump policy {args.policy}")
    if args.decimation is None:
        # The jump run was trained at 50 Hz (IsaacLab dt=0.005, decimation=4).
        # Keep the existing profiles at their original 100 Hz default.
        args.decimation = 20 if args.profile == "jump" else 10
    if args.vx is None:
        # Wait in place for the R trigger; preserve the prior default elsewhere.
        args.vx = 0.0 if args.profile == "jump" else 0.4

    env_yaml = args.env_yaml
    if env_yaml is None and not args.no_auto_env_yaml:
        env_yaml = infer_env_yaml(args.policy)
    control_cfg = load_control_config(env_yaml)
    if control_cfg.source is not None:
        print(f"[sim2sim] Loaded control/action config from {control_cfg.source}")
        print(f"[sim2sim] default_dof_pos={np.array2string(control_cfg.default_dof_pos, precision=3)}")
        print(f"[sim2sim] kp={np.array2string(control_cfg.kp, precision=3)}")
        print(f"[sim2sim] kd={np.array2string(control_cfg.kd, precision=3)}")
        print(f"[sim2sim] torque_limit={np.array2string(control_cfg.torque_limit, precision=3)}")
        print(f"[sim2sim] action_scale={np.array2string(control_cfg.action_scale, precision=3)}")
        print(f"[sim2sim] action_mode={control_cfg.action_mode.tolist()}")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.opt.timestep = args.dt
    if args.disable_scene_obstacles:
        disable_scene_obstacles(model)
    data = mujoco.MjData(model)

    qpos_adrs, qvel_adrs = joint_addresses(model)
    reset_pose(control_cfg, model, data, qpos_adrs, qvel_adrs)

    policy = None
    device = torch.device(args.device)
    if args.policy is not None:
        policy = torch.jit.load(str(args.policy), map_location=device)
        policy.eval()

    q, dq, omega, projected_gravity, _ = get_robot_state(model, data, qpos_adrs, qvel_adrs)
    command_state = {
        "vx": args.vx,
        "vy": args.vy,
        "wz": args.wz,
        "jump_active": args.jump_cmd >= 0.5,
    }

    def handle_key(key: str) -> None:
        key = key.upper()
        if key == "W":
            command_state["vx"] += args.keyboard_step
        elif key == "S":
            command_state["vx"] -= args.keyboard_step
        elif key == "A":
            command_state["vy"] += args.keyboard_step
        elif key == "D":
            command_state["vy"] -= args.keyboard_step
        elif key == "Q":
            command_state["wz"] += args.keyboard_step
        elif key == "E":
            command_state["wz"] -= args.keyboard_step
        elif key == "R" and args.profile == "jump":
            command_state["jump_active"] = not command_state["jump_active"]
        else:
            return

        jump_status = ""
        if args.profile == "jump":
            jump_status = f", jump_cmd={'ON' if command_state['jump_active'] else 'OFF'}"
        print(
            f"[sim2sim] vx={command_state['vx']:+.2f}, "
            f"vy={command_state['vy']:+.2f}, "
            f"wz={command_state['wz']:+.2f}{jump_status}"
        )

    def key_callback(keycode: int) -> None:
        if keycode in (ord("W"), ord("A"), ord("S"), ord("D"), ord("Q"), ord("E"), ord("R")):
            handle_key(chr(keycode))

    def poll_terminal_keys() -> None:
        stdin_fd = sys.stdin.fileno()
        while select.select([stdin_fd], [], [], 0.0)[0]:
            for key in os.read(stdin_fd, 128).decode(errors="ignore"):
                if key == "\x03":
                    raise KeyboardInterrupt
                handle_key(key)

    command = Command(
        command_state["vx"],
        command_state["vy"],
        command_state["wz"],
        args.left_leg_length,
        args.right_leg_length,
        args.tsk,
        float(command_state["jump_active"]),
    )
    action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    obs = build_observation(control_cfg, args.profile, command, q, dq, omega, projected_gravity, action)
    history = deque((obs.copy() for _ in range(args.history_len)), maxlen=args.history_len)

    viewer = None
    if not args.headless:
        mujoco_viewer = importlib.import_module("mujoco.viewer")

        viewer = mujoco_viewer.launch_passive(model, data, key_callback=key_callback)
        viewer.cam.trackbodyid = name_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -15.0
        controls = (
            f"[sim2sim] W/S: vx +/-{args.keyboard_step:g}, "
            f"A/D: vy +/-{args.keyboard_step:g}, "
            f"Q/E: wz +/-{args.keyboard_step:g}"
        )
        print(controls + (", R: toggle jump" if args.profile == "jump" else ""))

    terminal_settings = None
    if viewer is not None:
        if sys.stdin.isatty():
            terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            print("[sim2sim] Terminal keyboard control active; keep this terminal focused.")
        else:
            print("[sim2sim] stdin is not a TTY; use the MuJoCo window for keyboard control.")

    n_steps = int(args.duration / args.dt)
    control_steps = 0
    wall_start = time.time()
    try:
        for step in range(n_steps):
            q, dq, omega, projected_gravity, quat = get_robot_state(model, data, qpos_adrs, qvel_adrs)

            if step % args.decimation == 0:
                if terminal_settings is not None:
                    poll_terminal_keys()
                command = Command(
                    command_state["vx"],
                    command_state["vy"],
                    command_state["wz"],
                    args.left_leg_length,
                    args.right_leg_length,
                    args.tsk,
                    float(command_state["jump_active"]),
                )
                obs = build_observation(control_cfg, args.profile, command, q, dq, omega, projected_gravity, action)
                history.append(obs.copy())
                if policy is not None:
                    action = run_policy(policy, obs, history, device)
                    action = np.clip(action, -args.clip_actions, args.clip_actions)
                control_steps += 1

            data.ctrl[:] = pd_control(control_cfg, action, q, dq)
            mujoco.mj_step(model, data)

            if viewer is not None:
                if not viewer.is_running():
                    break
                if step % args.render_every == 0:
                    viewer.sync()

            # if args.print_every > 0 and step % args.print_every == 0:
            #     euler = quat_wxyz_to_euler_xyz(quat)
            #     jump_status = f"jump_cmd={command.jump_cmd:.0f} " if args.profile == "jump" else ""
            #     print(
            #         f"t={data.time:6.3f}s ctrl={control_steps:5d} "
            #         f"{jump_status}"
            #         f"base_z={data.qpos[2]:+.3f} roll={euler[0]:+.2f} pitch={euler[1]:+.2f} "
            #         f"action_norm={np.linalg.norm(action):.3f}"
            #     )

            if args.real_time and viewer is not None:
                target = wall_start + data.time
                delay = target - time.time()
                if delay > 0.0:
                    time.sleep(delay)
    finally:
        if terminal_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, terminal_settings)
        if viewer is not None:
            viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tita MuJoCo Python sim2sim demo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo scene.xml path.")
    parser.add_argument("--policy", type=Path, default=None, help="Optional TorchScript policy.pt.")
    parser.add_argument("--env-yaml", type=Path, default=None, help="Optional saved params/env.yaml to load control config.")
    parser.add_argument("--no-auto-env-yaml", action="store_true", help="Do not infer params/env.yaml from --policy.")
    parser.add_argument("--profile", choices=("command_gated", "velocity", "jump"), default="command_gated")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument(
        "--decimation",
        type=int,
        default=None,
        help="Policy/control update interval (default: jump=20, other profiles=10).",
    )
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--disable-scene-obstacles", action="store_true")
    parser.add_argument("--real-time", action="store_true", default=True)
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=1000, help="0 disables status prints.")
    parser.add_argument("--clip-actions", type=float, default=100.0)
    parser.add_argument("--vx", type=float, default=None, help="Forward command (default: jump=0.0, others=0.4).")
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--left-leg-length", type=float, default=0.5)
    parser.add_argument("--right-leg-length", type=float, default=0.5)
    parser.add_argument("--tsk", type=float, default=0.0)
    parser.add_argument("--keyboard-step", type=float, default=0.2, help="W/S/A/D command increment per key press.")
    parser.add_argument(
        "--jump-cmd",
        type=float,
        choices=(0.0, 1.0),
        default=0.0,
        help="Initial jump command for --profile jump; press R in the viewer to toggle it.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    simulate(parse_args())
