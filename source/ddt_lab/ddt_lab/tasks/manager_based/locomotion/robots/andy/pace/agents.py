"""NP3O runner configurations for Andy PACE tasks."""
from ..agents.np3o_cfg import andy_height_np3o_runner_cfg

def andy_pace_height_np3o_runner_cfg() -> dict:
    """Separate experiment directory for the nominal PACE motor model."""
    cfg = andy_height_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "andy_height_pace"
    return cfg


def andy_pace_robust_height_np3o_runner_cfg() -> dict:
    cfg = andy_pace_height_np3o_runner_cfg()
    cfg['runner']['experiment_name'] = 'andy_height_pace_robust'
    return cfg


def andy_pace_fixed_height_np3o_runner_cfg() -> dict:
    cfg = andy_pace_robust_height_np3o_runner_cfg()
    cfg['runner']['experiment_name'] = 'andy_pace_fixed035'
    cfg['runner']['max_iterations'] = 20000
    return cfg
