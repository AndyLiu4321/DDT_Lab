# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared terrain features for simulated ray scans and deployed depth cameras."""

from __future__ import annotations

import torch


def grid_height_profile(
    ray_hits_w: torch.Tensor,
    num_longitudinal: int = 17,
    num_lateral: int = 11,
    reference_start_column: int = 6,
    reference_end_column: int = 9,
) -> torch.Tensor:
    """Compress a 2-D height grid into a terrain-relative longitudinal profile.

    Isaac Lab's ``GridPatternCfg(ordering="xy")`` stores the 17 longitudinal
    samples as the inner dimension and the 11 lateral samples as the outer
    dimension.  Taking the lateral median rejects isolated bad rays and makes
    the feature straightforward to reproduce from a depth-camera point cloud.

    Returns:
        Tensor of shape ``(num_envs, num_longitudinal)`` in metres.  Zero is
        the locally estimated support surface; positive values are obstacles.
    """
    expected_rays = num_longitudinal * num_lateral
    if ray_hits_w.shape[1] != expected_rays:
        raise ValueError(
            f"Expected {expected_rays} grid rays "
            f"({num_lateral}x{num_longitudinal}), got {ray_hits_w.shape[1]}."
        )
    if not 0 <= reference_start_column < reference_end_column <= num_longitudinal:
        raise ValueError(
            "Reference column range must satisfy "
            f"0 <= start < end <= {num_longitudinal}, got "
            f"[{reference_start_column}, {reference_end_column})."
        )

    heights = ray_hits_w[..., 2].reshape(-1, num_lateral, num_longitudinal)
    heights = torch.where(
        torch.isfinite(heights),
        heights,
        torch.full_like(heights, torch.nan),
    )
    # One obstacle must be present across most of the robot width before it is
    # represented in the profile.
    longitudinal = torch.nanmedian(heights, dim=1).values
    # Use the three near-body columns x=[-0.2, -0.1, 0.0] as the support
    # surface.  Far-behind rays may lie on lower stairs and would bias the
    # reference downward as the robot climbs.
    reference = torch.nanmedian(
        longitudinal[:, reference_start_column:reference_end_column], dim=1
    ).values
    relative = longitudinal - reference.unsqueeze(1)
    return torch.nan_to_num(relative, nan=0.0, posinf=0.0, neginf=0.0)


def forward_height_profile(
    ray_hits_w: torch.Tensor,
    num_longitudinal: int = 17,
    num_lateral: int = 11,
    reference_start_column: int = 6,
    reference_end_column: int = 9,
    forward_start_column: int = 8,
) -> torch.Tensor:
    """Return only the camera-reproducible forward part of the profile."""
    if not 0 <= forward_start_column < num_longitudinal:
        raise ValueError(
            f"forward_start_column must be within [0, {num_longitudinal - 1}], "
            f"got {forward_start_column}."
        )
    profile = grid_height_profile(
        ray_hits_w,
        num_longitudinal=num_longitudinal,
        num_lateral=num_lateral,
        reference_start_column=reference_start_column,
        reference_end_column=reference_end_column,
    )
    return profile[:, forward_start_column:]


def depth_pointcloud_height_profile(
    points_b: torch.Tensor,
    support_height_b: torch.Tensor,
    num_bins: int = 9,
    grid_resolution: float = 0.1,
    lateral_half_width: float = 0.5,
    reduction: str = "median",
) -> torch.Tensor:
    """Convert a base-frame depth point cloud to the same 9-D Actor feature.

    Args:
        points_b: Point cloud with shape ``(batch, points, 3)`` in the robot
            base frame (x forward, y left, z up).
        support_height_b: Estimated support-plane z coordinate in the same
            frame, shape ``(batch,)``.  Deployment should obtain this from a
            local ground-plane fit or the robot's base-height estimator.
        num_bins: Number of forward longitudinal bins.
        grid_resolution: Distance between bin centres in metres.
        lateral_half_width: Keep points within this lateral corridor.
        reduction: Height reduction within each longitudinal bin. ``"median"``
            rejects sparse depth outliers, while ``"max"`` preserves the top
            edge of a vertical stair riser.

    Returns:
        Relative terrain heights at ``x=[0.0, 0.1, ..., 0.8]`` by default,
        shape ``(batch, num_bins)``.  Empty bins are zero and should be handled
        with temporal filtering/validity checks in the deployment node.
    """
    if points_b.ndim != 3 or points_b.shape[-1] != 3:
        raise ValueError(f"points_b must have shape (B, N, 3), got {tuple(points_b.shape)}.")
    if support_height_b.shape != (points_b.shape[0],):
        raise ValueError(
            f"support_height_b must have shape ({points_b.shape[0]},), "
            f"got {tuple(support_height_b.shape)}."
        )
    if reduction not in {"median", "max"}:
        raise ValueError(f"reduction must be 'median' or 'max', got {reduction!r}.")

    x, y, z = points_b.unbind(dim=-1)
    finite = torch.isfinite(points_b).all(dim=-1)
    lateral = torch.abs(y) <= lateral_half_width
    output = torch.zeros(
        points_b.shape[0],
        num_bins,
        device=points_b.device,
        dtype=points_b.dtype,
    )
    half_bin = 0.5 * grid_resolution
    for index in range(num_bins):
        centre = index * grid_resolution
        in_bin = finite & lateral & (x >= centre - half_bin) & (x < centre + half_bin)
        if reduction == "median":
            samples = torch.where(in_bin, z, torch.full_like(z, torch.nan))
            height = torch.nanmedian(samples, dim=1).values - support_height_b
            output[:, index] = torch.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            samples = torch.where(in_bin, z, torch.full_like(z, -torch.inf))
            height = torch.max(samples, dim=1).values - support_height_b
            has_samples = torch.any(in_bin, dim=1)
            output[:, index] = torch.where(has_samples, height, torch.zeros_like(height))
    return output
