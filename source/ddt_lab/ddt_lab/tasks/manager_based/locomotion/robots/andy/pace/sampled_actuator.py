"""200 Hz sampled PD with held torque on a 1 kHz physics clock."""
import torch
from isaaclab.actuators import DCMotor
from isaaclab.utils import configclass
from .actuator import PaceDCMotor
from .actuator_cfg import PaceDCMotorCfg


class SampledPaceDCMotor(PaceDCMotor):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._phase = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._held = torch.zeros_like(self.computed_effort)
        self._raw_held = torch.zeros_like(self.computed_effort)
        self._strength = torch.ones((self._num_envs, 1), device=self._device)

    def reset(self, env_ids):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._phase[ids] = 0
        self._held[ids] = 0
        self._raw_held[ids] = 0
        self._strength[ids] = torch.empty_like(self._strength[ids]).uniform_(*self.cfg.strength_range)
        count = self._phase[ids].numel()
        lag = torch.randint(0, self.cfg.max_delay + 1, (count,), device=self._device, dtype=torch.int)
        self.update_time_lags(lag, env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        # Evaluate the motor law, but latch its output only on a PD tick.
        # Per-environment phase allows a reset to apply a fresh command immediately.
        result = DCMotor.compute(self, control_action, joint_pos - self.encoder_bias, joint_vel)
        tick = (self._phase == 0)[:, None]
        self._raw_held = torch.where(tick, self.computed_effort, self._raw_held)
        self._held = torch.where(tick, result.joint_efforts * self._strength, self._held)
        self.computed_effort[:] = self._raw_held
        self.applied_effort[:] = self.torques_delay_buffer.compute(self._held)
        result.joint_efforts = self.applied_effort
        self._phase = (self._phase + 1) % self.cfg.control_decimation
        return result


@configclass
class SampledPaceDCMotorCfg(PaceDCMotorCfg):
    class_type: type = SampledPaceDCMotor
    control_decimation: int = 5
    strength_range: tuple[float, float] = (.8, 1.0)
