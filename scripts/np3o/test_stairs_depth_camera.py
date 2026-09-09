#!/usr/bin/env python3
"""Smoke-test Mini stairs depth perception and save one depth image."""

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="DDT-Stairs-Jump-Mini-Play-v0",
    help="Mini stairs-jump task to inspect.",
)
parser.add_argument(
    "--output",
    default="logs/depth_camera_test.png",
    help="Path for the normalized depth preview.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.headless:
    os.environ.pop("DISPLAY", None)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import ddt_lab.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from isaaclab.utils import math as math_utils

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp


def main():
    spec = gym.spec(args_cli.task)
    env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
    env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain.terrain_generator.num_cols = 1
    # Put the robot close enough to the first stair that both sensors must see
    # a non-flat profile during this smoke test.
    env_cfg.events.reset_base.params["pose_range"]["x"] = (0.55, 0.55)
    env_cfg.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)
    env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped
    env.reset()
    actions = torch.zeros(
        (unwrapped.num_envs, unwrapped.action_manager.total_action_dim),
        device=unwrapped.device,
    )
    for _ in range(12):
        env.step(actions)

    camera = unwrapped.scene["depth_camera"]
    depth = camera.data.output["depth"][0, ..., 0]
    valid = torch.isfinite(depth) & (depth > 0.08) & (depth < 3.0)
    # Play mode disables observation corruption, so this is exactly the 9-D
    # terrain_profile passed to the Actor.  Do not slice the final policy
    # dimensions: gait_phase now follows terrain_profile.
    depth_profile = mdp.depth_camera_height_profile(unwrapped)[0]
    scanner_profile = mdp.terrain_height_profile(unwrapped)[0]

    points_c = math_utils.unproject_depth(depth, camera.data.intrinsic_matrices[0])
    points_mount = torch.stack(
        (points_c[..., 2], -points_c[..., 0], -points_c[..., 1]),
        dim=-1,
    )
    mount_quat = torch.tensor(
        (0.953717, 0.0, 0.300706, 0.0),
        device=points_c.device,
    )
    mount_rot_b = math_utils.matrix_from_quat(mount_quat)
    points_b = points_mount @ mount_rot_b.T
    points_b += torch.tensor((0.27, 0.0, 0.04), device=points_c.device)
    robot = unwrapped.scene["robot"]
    root_rot_w = math_utils.matrix_from_quat(robot.data.root_link_quat_w[0])
    yaw_rot_w = math_utils.matrix_from_quat(
        math_utils.yaw_quat(robot.data.root_link_quat_w[0:1])
    )[0]
    points_b = points_b @ root_rot_w.T @ yaw_rot_w

    valid_depth = depth[valid]
    print(f"depth shape: {tuple(depth.shape)}", flush=True)
    print(f"valid pixels: {int(valid.sum())}/{depth.numel()}", flush=True)
    if valid_depth.numel() > 0:
        print(
            "valid depth range: "
            f"{float(valid_depth.min()):.3f} .. {float(valid_depth.max()):.3f} m",
            flush=True,
        )
    print(f"depth Actor profile: {depth_profile.detach().cpu().tolist()}", flush=True)
    print(f"scanner reference:  {scanner_profile.detach().cpu().tolist()}", flush=True)
    print(
        "point cloud base ranges: "
        f"x={float(points_b[torch.isfinite(points_b[:, 0]), 0].min()):.3f}.."
        f"{float(points_b[torch.isfinite(points_b[:, 0]), 0].max()):.3f}, "
        f"y={float(points_b[torch.isfinite(points_b[:, 1]), 1].min()):.3f}.."
        f"{float(points_b[torch.isfinite(points_b[:, 1]), 1].max()):.3f}, "
        f"z={float(points_b[torch.isfinite(points_b[:, 2]), 2].min()):.3f}.."
        f"{float(points_b[torch.isfinite(points_b[:, 2]), 2].max()):.3f}",
        flush=True,
    )
    print(f"robot root height: {float(robot.data.root_link_pos_w[0, 2]):.3f} m", flush=True)

    preview = torch.zeros_like(depth)
    if valid_depth.numel() > 0:
        span = (valid_depth.max() - valid_depth.min()).clamp_min(1.0e-6)
        preview[valid] = 1.0 - (valid_depth - valid_depth.min()) / span
    preview_u8 = (preview * 255.0).byte().cpu().numpy()
    preview_u8 = np.repeat(preview_u8[..., None], 3, axis=-1)
    output_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(preview_u8).resize((480, 320), Image.Resampling.NEAREST).save(output_path)
    print(f"saved depth preview: {output_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
