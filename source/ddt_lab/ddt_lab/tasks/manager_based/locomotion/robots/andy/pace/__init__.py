"""Andy PACE tasks, motor model, identification data and robustness curricula.

Legacy module aliases preserve imports used in historical env.yaml/pickle files
without retaining duplicate implementation files in the parent Andy directory.
"""
import sys
from importlib import import_module
import gymnasium as gym

_LEGACY_MODULES = {
    'pace_actuator': 'actuator',
    'pace_actuator_cfg': 'actuator_cfg',
    'pace_height_env_cfg': 'height_env_cfg',
    'pace_robust_env_cfg': 'robust_env_cfg',
    'pace_robust_mdp': 'robust_mdp',
    'pace_stationary_mdp': 'stationary_mdp',
}
_parent_name = __name__.rsplit('.', 1)[0]
for _old, _new in _LEGACY_MODULES.items():
    _module = import_module(f'{__name__}.{_new}')
    sys.modules[f'{_parent_name}.{_old}'] = _module
    setattr(sys.modules[_parent_name], _old, _module)

from .height_env_cfg import AndyPaceHeightFlatEnvCfg, AndyPaceHeightFlatEnvCfg_PLAY
from .robust_env_cfg import AndyPaceRobustHeightFlatEnvCfg, AndyPaceRobustHeightFlatEnvCfg_PLAY

for _task, _cfg, _runner in (
    ('DDT-Height-Flat-Andy-Pace-v0', AndyPaceHeightFlatEnvCfg, 'andy_pace_height_np3o_runner_cfg'),
    ('DDT-Height-Flat-Andy-Pace-Play-v0', AndyPaceHeightFlatEnvCfg_PLAY, 'andy_pace_height_np3o_runner_cfg'),
    ('DDT-Height-Flat-Andy-Pace-Robust-v0', AndyPaceRobustHeightFlatEnvCfg, 'andy_pace_robust_height_np3o_runner_cfg'),
    ('DDT-Height-Flat-Andy-Pace-Robust-Play-v0', AndyPaceRobustHeightFlatEnvCfg_PLAY, 'andy_pace_robust_height_np3o_runner_cfg'),
):
    gym.register(id=_task, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': _cfg,
                         'np3o_cfg_entry_point': f'{__name__}.agents:{_runner}'})

from .flat_env_cfg import AndyPaceFixedHeightFlatEnvCfg, AndyPaceFixedHeightFlatEnvCfg_PLAY

for _task, _cfg in (
    ('DDT-Velocity-Flat-Andy-Pace-Fixed035-v0', AndyPaceFixedHeightFlatEnvCfg),
    ('DDT-Velocity-Flat-Andy-Pace-Fixed035-Play-v0', AndyPaceFixedHeightFlatEnvCfg_PLAY),
):
    gym.register(id=_task, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': _cfg,
                         'np3o_cfg_entry_point': f'{__name__}.agents:andy_pace_fixed_height_np3o_runner_cfg'})
