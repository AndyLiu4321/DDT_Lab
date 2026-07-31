#!/usr/bin/env python3
"""MuJoCo Python sim2sim demo for the D1 rough velocity policy.

This script targets ``DDT-Velocity-Rough-D1-Play-v0`` and the D1 MuJoCo
description under ``ddt_ros2_control/urdfs/d1_description/mujoco``.

The actor observation layout follows the exported NP3O rough D1 policy:

    base_ang_vel(3), projected_gravity(3), base_velocity_command(3),
    joint_pos_rel_without_wheel(16), joint_vel(16), last_action(16)

Examples:
    python scripts/sim2sim/d1_rough_mujoco_demo.py --headless --duration 5

    python scripts/sim2sim/d1_rough_mujoco_demo.py \
        --vx 0.4 --heading 0.0 --disable-scene-obstacles
"""

from __future__ import annotations

import argparse
import importlib
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for user setup.
    raise SystemExit("Python package 'mujoco' is not installed. Run: python -m pip install mujoco") from exc

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for user setup.
    raise SystemExit("Python package 'torch' is not installed. Use the isaaclab conda environment.") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "ddt_ros2_control/urdfs/d1_description/mujoco/scene.xml"
DEFAULT_POLICY = REPO_ROOT / "logs/np3o/d1_rough/2026-06-10_18-01-40/exported/policy.pt"
OBS_DIM = 57

JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FL_foot_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FR_foot_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RL_foot_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RR_foot_joint",
)
WHEEL_JOINT_INDICES = np.array([3, 7, 11, 15], dtype=np.int64)

DEFAULT_DOF_POS = np.array(
    [
        0.0,
        0.8,
        -1.5,
        -1.5,
        0.0,
        0.8,
        -1.5,
        -1.5,
        0.0,
        0.8,
        -1.5,
        -1.5,
        0.0,
        0.8,
        -1.5,
        -1.5,
    ],
    dtype=np.float64,
)
DEFAULT_KP = np.array([60.0, 60.0, 60.0, 0.0] * 4, dtype=np.float64)
DEFAULT_KD = np.array([1.5, 1.5, 1.5, 0.5] * 4, dtype=np.float64)
DEFAULT_TORQUE_LIMIT = np.array([60.0, 60.0, 60.0, 12.0] * 4, dtype=np.float64)
DEFAULT_ACTION_SCALE = np.array([0.25, 0.25, 0.25, 5.0] * 4, dtype=np.float64)
DEFAULT_ACTION_MODE = np.array(["pos", "pos", "pos", "vel"] * 4, dtype=object)


@dataclass(frozen=True)
class Command:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


@dataclass
class ControlConfig:
    default_dof_pos: np.ndarray = field(default_factory=lambda: DEFAULT_DOF_POS.copy())
    root_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.6], dtype=np.float64))
    root_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    kp: np.ndarray = field(default_factory=lambda: DEFAULT_KP.copy())
    kd: np.ndarray = field(default_factory=lambda: DEFAULT_KD.copy())
    torque_limit: np.ndarray = field(default_factory=lambda: DEFAULT_TORQUE_LIMIT.copy())
    action_scale: np.ndarray = field(default_factory=lambda: DEFAULT_ACTION_SCALE.copy())
    action_mode: np.ndarray = field(default_factory=lambda: DEFAULT_ACTION_MODE.copy())
    heading_control_stiffness: float = 0.5
    yaw_rate_range: tuple[float, float] = (-1.0, 1.0)
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


def scale_for_joint(scale, joint_name: str) -> float | None:
    if scale is None:
        return None
    if isinstance(scale, (float, int)):
        return float(scale)
    if isinstance(scale, dict):
        for pattern, value in scale.items():
            if joint_regex_matches(str(pattern), joint_name):
                return float(value)
    return None


def load_control_config(env_yaml: Path | None) -> ControlConfig:
    cfg = ControlConfig(source=env_yaml)
    if env_yaml is None:
        return cfg

    with env_yaml.open("r", encoding="utf-8") as f:
        env_cfg = yaml.unsafe_load(f)

    robot_cfg = env_cfg.get("scene", {}).get("robot", {})
    init_state = robot_cfg.get("init_state", {})
    joint_pos = init_state.get("joint_pos", {})
    if init_state.get("pos") is not None:
        cfg.root_pos = np.asarray(init_state["pos"], dtype=np.float64)
    if init_state.get("rot") is not None:
        cfg.root_quat = np.asarray(init_state["rot"], dtype=np.float64)

    for pattern, value in joint_pos.items():
        for i, name in enumerate(JOINT_NAMES):
            if joint_regex_matches(str(pattern), name):
                cfg.default_dof_pos[i] = float(value)

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
            if any(joint_regex_matches(str(pattern), name) for pattern in patterns):
                if stiffness is not None:
                    cfg.kp[i] = float(stiffness)
                if damping is not None:
                    cfg.kd[i] = float(damping)
                if effort_limit is not None:
                    cfg.torque_limit[i] = float(effort_limit)

    cfg.action_scale[:] = 0.0
    for action in env_cfg.get("actions", {}).values():
        joint_names = action.get("joint_names", [])
        scale = action.get("scale")
        class_type = str(action.get("class_type", ""))
        mode = "vel" if "JointVelocityAction" in class_type else "pos"
        for pattern in joint_names:
            for i, name in enumerate(JOINT_NAMES):
                if joint_regex_matches(str(pattern), name):
                    value = scale_for_joint(scale, name)
                    if value is not None:
                        cfg.action_scale[i] = value
                        cfg.action_mode[i] = mode

    missing = cfg.action_scale == 0.0
    cfg.action_scale[missing] = DEFAULT_ACTION_SCALE[missing]
    cfg.action_mode[missing] = DEFAULT_ACTION_MODE[missing]

    command_cfg = env_cfg.get("commands", {}).get("base_velocity", {})
    cfg.heading_control_stiffness = float(command_cfg.get("heading_control_stiffness", cfg.heading_control_stiffness))
    ranges = command_cfg.get("ranges", {})
    if ranges.get("ang_vel_z") is not None:
        cfg.yaw_rate_range = tuple(float(x) for x in ranges["ang_vel_z"])
    return cfg


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
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
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


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
    data: mujoco.MjData, qpos_adrs: np.ndarray, qvel_adrs: np.ndarray
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
    joint_pos_rel = q - cfg.default_dof_pos
    joint_pos_rel[WHEEL_JOINT_INDICES] = 0.0

    obs[0:3] = omega * 0.25
    obs[3:6] = projected_gravity
    obs[6:9] = np.array([command.vx * 2.0, command.vy * 2.0, command.wz * 0.25], dtype=np.float32)
    obs[9:25] = joint_pos_rel
    obs[25:41] = dq * 0.05
    obs[41:57] = last_action
    return obs


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
        action = policy(current_t, history_t)
    return action.detach().cpu().numpy()[0].astype(np.float64)


def reset_pose(cfg: ControlConfig, model: mujoco.MjModel, data: mujoco.MjData,
               qpos_adrs: np.ndarray) -> None:
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


def maybe_policy_dim(policy: torch.jit.ScriptModule) -> int | None:
    for name, buf in policy.named_buffers():
        if name == "backbone.obs_normalizer._mean" and len(buf.shape) == 2:
            return int(buf.shape[1])
    return None


def command_from_args(args: argparse.Namespace, cfg: ControlConfig, quat: np.ndarray) -> Command:
    wz = args.wz
    if args.heading is not None:
        yaw = quat_wxyz_to_euler_xyz(quat)[2]
        wz = cfg.heading_control_stiffness * wrap_to_pi(args.heading - yaw)
        wz = float(np.clip(wz, cfg.yaw_rate_range[0], cfg.yaw_rate_range[1]))
    return Command(args.vx, args.vy, wz)


def simulate(args: argparse.Namespace) -> None:
    env_yaml = args.env_yaml
    if env_yaml is None and not args.no_auto_env_yaml:
        env_yaml = infer_env_yaml(args.policy)
    control_cfg = load_control_config(env_yaml)
    if control_cfg.source is not None:
        print(f"[d1_rough] Loaded control/action config from {control_cfg.source}")
        print(f"[d1_rough] default_dof_pos={np.array2string(control_cfg.default_dof_pos, precision=3)}")
        print(f"[d1_rough] kp={np.array2string(control_cfg.kp, precision=3)}")
        print(f"[d1_rough] kd={np.array2string(control_cfg.kd, precision=3)}")
        print(f"[d1_rough] torque_limit={np.array2string(control_cfg.torque_limit, precision=3)}")
        print(f"[d1_rough] action_scale={np.array2string(control_cfg.action_scale, precision=3)}")
        print(f"[d1_rough] action_mode={control_cfg.action_mode.tolist()}")

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
        raise RuntimeError(f"Expected D1 rough policy input dim {OBS_DIM}, got {policy_dim}.")

    q, dq, omega, projected_gravity, quat = get_robot_state(data, qpos_adrs, qvel_adrs)
    command = command_from_args(args, control_cfg, quat)
    action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    obs = build_observation(control_cfg, command, q, dq, omega, projected_gravity, action)
    history = deque((obs.copy() for _ in range(args.history_len)), maxlen=args.history_len)

    viewer = None
    if not args.headless:
        mujoco_viewer = importlib.import_module("mujoco.viewer")
        viewer = mujoco_viewer.launch_passive(model, data)
        viewer.cam.trackbodyid = name_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        viewer.cam.distance = 3.0
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
                command = command_from_args(args, control_cfg, quat)
                obs = build_observation(control_cfg, command, q, dq, omega, projected_gravity, action)
                history.append(obs.copy())
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

            if args.print_every > 0 and step % args.print_every == 0:
                euler = quat_wxyz_to_euler_xyz(quat)
                print(
                    f"t={data.time:6.3f}s ctrl={control_steps:5d} "
                    f"cmd=({command.vx:+.2f},{command.vy:+.2f},{command.wz:+.2f}) "
                    f"base_z={data.qpos[2]:+.3f} z_range=({min_base_z:+.3f},{max_base_z:+.3f}) "
                    f"roll={euler[0]:+.2f} pitch={euler[1]:+.2f} yaw={euler[2]:+.2f} "
                    f"action_norm={np.linalg.norm(action):.3f}"
                )

            if args.real_time and viewer is not None:
                delay = wall_start + data.time - time.time()
                if delay > 0.0:
                    time.sleep(delay)
    finally:
        if viewer is not None:
            viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D1 rough MuJoCo Python sim2sim demo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo d1 scene.xml path.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="TorchScript D1 rough policy.pt.")
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
    parser.add_argument(
        "--heading",
        type=float,
        default=None,
        help="Optional target world yaw. If set, wz follows D1's heading_command proportional controller.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    simulate(parse_args())
