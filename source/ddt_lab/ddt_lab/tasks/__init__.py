# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
# AndyMini is an Isaac Gym reference, not an Isaac Lab task package.
# The importer matches substrings, so this also excludes renamed copies such as 1andymini.
_BLACKLIST_PKGS = ["utils", ".mdp", "andymini"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
