#!/usr/bin/env python
"""Test script to verify Mini robot environment registration."""

import gymnasium as gym

# List all available environments related to Mini
print("Registered Mini environments:")
print("-" * 60)

all_envs = gym.envs.registry.keys()
mini_envs = [env for env in all_envs if "Mini" in env]

if mini_envs:
    for env_id in sorted(mini_envs):
        print(f"  ✓ {env_id}")
else:
    print("  ✗ No Mini environments found!")

print("-" * 60)

# Try to instantiate one environment
try:
    print("\nAttempting to create DDT-Velocity-Flat-Mini-v0...")
    env = gym.make("DDT-Velocity-Flat-Mini-v0")
    print("✓ Environment created successfully!")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    env.close()
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
