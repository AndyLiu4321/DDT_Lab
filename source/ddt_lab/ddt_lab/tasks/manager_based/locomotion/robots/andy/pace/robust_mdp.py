"""Measured-performance push curriculum and unambiguous height logging."""
import torch
from isaaclab.envs import mdp
from ..height_mdp import HeightCommand, base_height


class RobustHeightCommand(HeightCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._height_sum = torch.zeros(self.num_envs, device=self.device)
        self._height_count = torch.zeros_like(self._height_sum)
        self.metrics['height_mae_m'] = torch.zeros_like(self._height_sum)

    def _update_metrics(self):
        super()._update_metrics()  # Preserve the historical metric for comparison.
        self._height_sum += (base_height(self._env) - self.command[:, 1]).abs()
        self._height_count += 1
        self.metrics['height_mae_m'][:] = self._height_sum / self._height_count.clamp_min(1)

    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._height_sum[ids] = 0
        self._height_count[ids] = 0
        return result


def push_curriculum(env, env_ids, min_steps=5000, min_episodes=1024):
    """Assess completed episodes before managers reset; never promote by time alone.

    Stages are additive XY velocity impulses: 0, .05, .10, .15, .20 m/s.
    A stage needs >=100 s simulated time and >=1024 completed episodes.
    Promotion: >=95% non-failure episodes AND normalized height reward >=.9.
    Demotion: survival <90% OR height reward <.8. Initial resets are excluded.
    Curriculum state deliberately restarts at zero when resuming a process.
    """
    state = getattr(env, '_pace_push_state', None)
    if state is None:
        state = env._pace_push_state = dict(stage=0, start=int(env.common_step_counter),
                                            count=0, survived=0.0, height=0.0,
                                            last_survival=0.0, last_height=0.0)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[env.episode_length_buf[ids] > 0]
    if ids.numel():
        duration = env.episode_length_buf[ids].float() * env.step_dt
        weight = env.reward_manager.get_term_cfg('tracking_base_height').weight
        height = env.reward_manager._episode_sums['tracking_base_height'][ids] / (duration * weight)
        state['count'] += ids.numel()
        state['survived'] += (~env.termination_manager.terminated[ids]).float().sum().item()
        state['height'] += height.sum().item()
    if int(env.common_step_counter) - state['start'] >= min_steps and state['count'] >= min_episodes:
        survival = state['survived'] / state['count']
        height = state['height'] / state['count']
        if survival >= .95 and height >= .9:
            state['stage'] = min(4, state['stage'] + 1)
        elif survival < .9 or height < .8:
            state['stage'] = max(0, state['stage'] - 1)
        state.update(start=int(env.common_step_counter), count=0, survived=0.0, height=0.0,
                     last_survival=survival, last_height=height)
    return {'stage': float(state['stage']), 'max_delta_v': .05 * state['stage'],
            'survival': state['last_survival'], 'height_quality': state['last_height']}


def progressive_push(env, env_ids):
    stage = getattr(env, '_pace_push_state', {}).get('stage', 0)
    if stage == 0:
        return
    amplitude = .05 * stage
    mdp.push_by_setting_velocity(env, env_ids, velocity_range={
        'x': (-amplitude, amplitude), 'y': (-amplitude, amplitude),
    })
