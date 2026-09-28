#!/usr/bin/env python3
"""MuJoCo Python sim2sim demo for the Mini jump policy.

This script targets the 6-DoF Mini/Tita6 model used by
``MiniJumpFlatEnvCfg`` and the exported NP3O policy from:

    logs/np3o/mini_jump/2026-07-03_11-42-49/exported/policy.pt

The policy observation layout follows ``jump_env_cfg.py``:

    base_ang_vel(3), projected_gravity(3), base_velocity_command(3),
    joint_pos_rel_without_wheel(6), joint_vel(6), last_action(6), jump_cmd(1)

Examples:
    python scripts/sim2sim/mini_jump_mujoco_demo.py --headless --duration 5

    python scripts/sim2sim/mini_jump_mujoco_demo.py \
        --policy logs/np3o/mini_jump/2026-07-03_11-42-49/exported/policy.pt \
        --jump-cmd 1 --vx 0 --vy 0 --wz 0 --disable-scene-obstacles
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
DEFAULT_MODEL = REPO_ROOT / "ddt_ros2_control/urdfs/tita6_description/mujoco/scene.xml"
DEFAULT_POLICY = REPO_ROOT / "logs/np3o/mini_jump/2026-07-03_11-42-49/exported/policy.pt"

JOINT_NAMES = (
    "joint_left_leg_2",
    "joint_left_leg_3",
    "joint_left_leg_4",
    "joint_right_leg_2",
    "joint_right_leg_3",
    "joint_right_leg_4",
)
# MuJoCo/action order is [L2, L3, L4, R2, R3, R4], while IsaacLab's
# articulation observations are [L2, R2, L3, R3, L4, R4].
OBS_JOINT_INDICES = np.array([0, 3, 1, 4, 2, 5], dtype=np.int64)
OBS_WHEEL_JOINT_INDICES = np.array([4, 5], dtype=np.int64)

DEFAULT_DOF_POS = np.array([0.8, -1.5, 0.0, 0.8, -1.5, 0.0], dtype=np.float64)
DEFAULT_KP = np.array([60.0, 60.0, 0.0, 60.0, 60.0, 0.0], dtype=np.float64)
DEFAULT_KD = np.array([1.5, 1.5, 0.5, 1.5, 1.5, 0.5], dtype=np.float64)
DEFAULT_TORQUE_LIMIT = np.array([60.0, 60.0, 12.0, 60.0, 60.0, 12.0], dtype=np.float64)
DEFAULT_ACTION_SCALE = np.array([0.25, 0.25, 5.0, 0.25, 0.25, 5.0], dtype=np.float64)
DEFAULT_ACTION_MODE = np.array(["pos", "pos", "vel", "pos", "pos", "vel"], dtype=object)


@dataclass(frozen=True)
class Command:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    jump_cmd: float = 0.0


@dataclass
class ControlConfig:
    default_dof_pos: np.ndarray = field(default_factory=lambda: DEFAULT_DOF_POS.copy())
    root_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.40], dtype=np.float64))
    root_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    kp: np.ndarray = field(default_factory=lambda: DEFAULT_KP.copy())
    kd: np.ndarray = field(default_factory=lambda: DEFAULT_KD.copy())
    torque_limit: np.ndarray = field(default_factory=lambda: DEFAULT_TORQUE_LIMIT.copy())
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
    if init_state.get("pos") is not None:
        cfg.root_pos = np.asarray(init_state["pos"], dtype=np.float64)
    if init_state.get("rot") is not None:
        cfg.root_quat = np.asarray(init_state["rot"], dtype=np.float64)
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
        scale = action.get("scale")
        class_type = str(action.get("class_type", ""))
        if scale is None:
            continue
        mode = "vel" if "JointVelocityAction" in class_type else "pos"
        for pattern in joint_names:
            for i, name in enumerate(JOINT_NAMES):
                if joint_regex_matches(pattern, name):
                    cfg.action_scale[i] = float(scale)
                    cfg.action_mode[i] = mode

    missing = cfg.action_scale == 0.0
    cfg.action_scale[missing] = DEFAULT_ACTION_SCALE[missing]
    cfg.action_mode[missing] = DEFAULT_ACTION_MODE[missing]
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
    obs = np.zeros(28, dtype=np.float32)
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
    obs[27] = command.jump_cmd
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


def simulate(args: argparse.Namespace) -> None:
    env_yaml = args.env_yaml
    if env_yaml is None and not args.no_auto_env_yaml:
        env_yaml = infer_env_yaml(args.policy)
    control_cfg = load_control_config(env_yaml)
    control_cfg.root_pos[2] = args.root_height
    if control_cfg.source is not None:
        print(f"[mini_jump] Loaded control/action config from {control_cfg.source}")
        print(f"[mini_jump] default_dof_pos={np.array2string(control_cfg.default_dof_pos, precision=3)}")
        print(f"[mini_jump] kp={np.array2string(control_cfg.kp, precision=3)}")
        print(f"[mini_jump] kd={np.array2string(control_cfg.kd, precision=3)}")
        print(f"[mini_jump] torque_limit={np.array2string(control_cfg.torque_limit, precision=3)}")
        print(f"[mini_jump] action_scale={np.array2string(control_cfg.action_scale, precision=3)}")
        print(f"[mini_jump] action_mode={control_cfg.action_mode.tolist()}")
    print(
        f"[mini_jump] policy_period={args.dt * args.decimation:.3f}s "
        f"({1.0 / (args.dt * args.decimation):.1f} Hz), root_height={args.root_height:.3f}m"
    )

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
    if policy_dim is not None and policy_dim != 28:
        raise RuntimeError(f"Expected mini_jump policy input dim 28, got {policy_dim}.")

    q, dq, omega, projected_gravity, _ = get_robot_state(data, qpos_adrs, qvel_adrs)
    command = Command(args.vx, args.vy, args.wz, args.jump_cmd)
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
    wall_start = time.time()
    try:
        for step in range(n_steps):
            q, dq, omega, projected_gravity, quat = get_robot_state(data, qpos_adrs, qvel_adrs)
            max_base_z = max(max_base_z, float(data.qpos[2]))

            if step % args.decimation == 0:
                if args.trigger_step >= 0:
                    active_jump = 1.0 if control_steps >= args.trigger_step else 0.0
                    command = Command(args.vx, args.vy, args.wz, active_jump)
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
                    f"t={data.time:6.3f}s ctrl={control_steps:5d} jump_cmd={command.jump_cmd:.0f} "
                    f"base_z={data.qpos[2]:+.3f} max_z={max_base_z:+.3f} "
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
    parser = argparse.ArgumentParser(description="Mini jump MuJoCo Python sim2sim demo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo tita6 scene.xml path.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="TorchScript mini_jump policy.pt.")
    parser.add_argument("--env-yaml", type=Path, default=None, help="Optional saved params/env.yaml.")
    parser.add_argument("--no-auto-env-yaml", action="store_true", help="Do not infer params/env.yaml from --policy.")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--decimation", type=int, default=20)
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--root-height", type=float, default=0.35, help="Initial base height used by the Play task.")
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
    parser.add_argument("--jump-cmd", type=float, default=1.0, help="Constant jump command when --trigger-step < 0.")
    parser.add_argument(
        "--trigger-step",
        type=int,
        default=-1,
        help="Policy control step to switch jump_cmd from 0 to 1. Default keeps --jump-cmd constant.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    simulate(parse_args())
