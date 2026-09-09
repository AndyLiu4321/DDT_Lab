# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster, TiledCamera
from isaaclab.utils import math as math_utils

from .terrain_features import depth_pointcloud_height_profile, forward_height_profile

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos - asset.data.default_joint_pos
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    joint_pos_rel = joint_pos_rel[:, asset_cfg.joint_ids]
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def biped_phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    """Per-foot gait phase clock for alternating biped gait.

    Returns 4 values: [sin_L, cos_L, sin_R, cos_R].
    Left and right feet are half a cycle apart (phase offset = π).
    """
    t = env.episode_length_buf.float() * env.step_dt  # (B,)
    phi_L = 2.0 * torch.pi * t / cycle_time            # left foot phase
    phi_R = phi_L + torch.pi                            # right foot: half cycle offset
    return torch.stack(
        [torch.sin(phi_L), torch.cos(phi_L), torch.sin(phi_R), torch.cos(phi_R)], dim=-1
    )


def terrain_height_profile(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    num_longitudinal: int = 17,
    num_lateral: int = 11,
    reference_start_column: int = 6,
    reference_end_column: int = 9,
    forward_start_column: int = 8,
) -> torch.Tensor:
    """Nine-bin forward terrain profile shared by scanner and depth deployment.

    With Mini's default 0.1 m grid this returns relative heights at
    ``x=[0.0, 0.1, ..., 0.8]`` metres.  A deployed depth-camera node should
    transform its point cloud to the base yaw frame, estimate the support
    plane, take a lateral median in the same bins, and supply these nine values
    in the same order.
    """
    sensor: RayCaster = env.scene[sensor_cfg.name]
    return forward_height_profile(
        sensor.data.ray_hits_w,
        num_longitudinal=num_longitudinal,
        num_lateral=num_lateral,
        reference_start_column=reference_start_column,
        reference_end_column=reference_end_column,
        forward_start_column=forward_start_column,
    )


def depth_camera_height_profile(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    data_type: str = "depth",
    mount_position_b: tuple[float, float, float] = (0.27, 0.0, 0.04),
    mount_orientation_b: tuple[float, float, float, float] = (0.953717, 0.0, 0.300706, 0.0),
    min_depth: float = 0.08,
    max_depth: float = 3.0,
    default_support_height: float = -0.35,
    support_x_min: float = 0.10,
    support_x_max: float = 0.55,
    support_lateral_half_width: float = 0.35,
    support_quantile: float = 0.10,
    num_bins: int = 9,
    grid_resolution: float = 0.1,
    lateral_half_width: float = 0.5,
) -> torch.Tensor:
    """Compress a base-mounted depth image into the Actor's 9-D profile.

    The camera depth image is unprojected with its actual intrinsics, transformed
    from ROS optical coordinates into the robot base frame, and reduced to
    relative terrain heights at ``x=[0.0, 0.1, ..., 0.8]`` metres.  The local
    support height is estimated only from the depth cloud; no height-scanner
    value enters this observation.
    """
    camera: TiledCamera = env.scene[sensor_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    camera_data = camera.data
    if data_type not in camera_data.output:
        raise KeyError(
            f"Depth camera '{sensor_cfg.name}' has no output '{data_type}'. "
            f"Available outputs: {tuple(camera_data.output.keys())}."
        )

    depth = camera_data.output[data_type]
    valid_depth = torch.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    depth = torch.where(valid_depth, depth, torch.full_like(depth, torch.nan))
    points_c = math_utils.unproject_depth(depth, camera_data.intrinsic_matrices)

    # unproject_depth returns ROS optical coordinates (x right, y down,
    # z forward).  First express them in the camera's world convention
    # (x forward, y left, z up), then apply the known rigid base_link mount.
    # Using fixed extrinsics avoids querying a moving XformPrimView for every
    # replicated environment and is identical to real-camera calibration.
    points_mount = torch.stack(
        (points_c[..., 2], -points_c[..., 0], -points_c[..., 1]),
        dim=-1,
    )
    mount_quat = torch.tensor(
        mount_orientation_b,
        device=points_c.device,
        dtype=points_c.dtype,
    ).unsqueeze(0)
    mount_rot_b = math_utils.matrix_from_quat(mount_quat)
    points_b = torch.matmul(points_mount, mount_rot_b[0].T)
    mount_pos_b = torch.tensor(
        mount_position_b,
        device=points_c.device,
        dtype=points_c.dtype,
    )
    points_b = points_b + mount_pos_b

    # Match RayCaster's attach_yaw_only=True frame.  The camera follows base
    # roll/pitch, so directly binning in base coordinates would turn a level
    # floor into a false slope whenever the robot leans.  IMU orientation is
    # available both in simulation and on the deployed robot.
    root_rot_w = math_utils.matrix_from_quat(robot.data.root_link_quat_w)
    yaw_rot_w = math_utils.matrix_from_quat(
        math_utils.yaw_quat(robot.data.root_link_quat_w)
    )
    points_w_rel = torch.bmm(points_b, root_rot_w.transpose(1, 2))
    points_b = torch.bmm(points_w_rel, yaw_rot_w)

    finite = torch.isfinite(points_b).all(dim=-1)
    support_mask = (
        finite
        & (points_b[..., 0] >= support_x_min)
        & (points_b[..., 0] <= support_x_max)
        & (torch.abs(points_b[..., 1]) <= support_lateral_half_width)
        & (points_b[..., 2] < 0.0)
    )
    support_samples = torch.where(
        support_mask,
        points_b[..., 2],
        torch.full_like(points_b[..., 2], torch.nan),
    )
    # Use the local lower envelope rather than the median.  Near a stair edge
    # the image contains the whole vertical riser, whose median would
    # underestimate the actual step height by roughly one half.
    support_height_b = torch.nanquantile(support_samples, support_quantile, dim=1)
    support_height_b = torch.nan_to_num(
        support_height_b,
        nan=default_support_height,
        posinf=default_support_height,
        neginf=default_support_height,
    )

    return depth_pointcloud_height_profile(
        points_b,
        support_height_b,
        num_bins=num_bins,
        grid_resolution=grid_resolution,
        lateral_half_width=lateral_half_width,
        # A maximum would amplify the robot's wheel edge or the next stair
        # riser.  Median matches the scanner's lateral-median feature and is
        # much more stable for real depth noise.
        reduction="median",
    )


# ---------------------------------------------------------------------------
# Privileged observation terms (critic-only)
# Mirrors the reference ``LocomotionWithNP3O`` priv_latent subset that can be
# obtained from Isaac Lab's Articulation data without extra PhysX API calls.
# ---------------------------------------------------------------------------


def contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary foot-contact state, centred at zero: +0.5 in contact, -0.5 not.

    Shape: ``(num_envs, num_feet)`` — typically 4 for D1.
    Mirrors reference ``contact_filt.float() - 0.5``.
    """
    from isaaclab.sensors import ContactSensor

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (B, K, 3)
    in_contact = (net_forces.norm(dim=-1) > threshold).float()  # (B, K)
    return in_contact - 0.5


def joint_kp_factor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Randomised kp (joint stiffness) as a scale factor relative to default.

    Shape: ``(num_envs, num_joints)``.
    Mirrors reference ``kp_factor`` in priv_latent; values outside [0, 2]
    are clamped to avoid very large signals from near-zero defaults.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    kp = asset.data.joint_stiffness[:, asset_cfg.joint_ids]
    default_kp = asset.data.default_joint_stiffness[:, asset_cfg.joint_ids]
    return (kp / (default_kp.abs() + 1e-6)).clamp(0.0, 2.0)


def joint_kd_factor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Randomised kd (joint damping) as a scale factor relative to default.

    Shape: ``(num_envs, num_joints)``.
    Mirrors reference ``kd_factor`` in priv_latent.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    kd = asset.data.joint_damping[:, asset_cfg.joint_ids]
    default_kd = asset.data.default_joint_damping[:, asset_cfg.joint_ids]
    return (kd / (default_kd.abs() + 1e-6)).clamp(0.0, 2.0)


def jump_stage(env: ManagerBasedRLEnv, command_name: str = "jump_cmd") -> torch.Tensor:
    """Normalized jump FSM stage for privileged observations."""
    term = env.command_manager.get_term(command_name)
    return (term.jump_stage.float() / float(term.STAGE_LAND)).unsqueeze(1)


def jump_state(env: ManagerBasedRLEnv, command_name: str = "jump_cmd") -> torch.Tensor:
    """Privileged jump FSM state: stage, was_in_flight, has_jumped, normalized max height."""
    term = env.command_manager.get_term(command_name)
    target_height = getattr(term.cfg, "target_height", 0.8)
    return torch.stack(
        [
            term.jump_stage.float() / float(term.STAGE_LAND),
            term.was_in_flight.float(),
            term.has_jumped.float(),
            term.max_height / max(target_height, 1.0e-6),
        ],
        dim=1,
    )
