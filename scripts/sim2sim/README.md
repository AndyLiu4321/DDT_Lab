# Sim2sim 脚本

两个 Andy 脚本不是重复实现，控制约定不同，因此分别保留：

| 入口 | 配置来源 | 用途 |
| --- | --- | --- |
| `andy_height_mujoco_demo.py` | 导出策略旁的 `params/env.yaml`（或 `--env-yaml`） | 按训练配置运行 MuJoCo 对照，PD 每个物理步更新 |
| `andy_height_mujoco_rl_sar.py` | `andy_rl_sar_reference/policy/andymini/robot_lab/config.yaml`（或 `--config`） | rl_sar 部署流程，独立的物理、PD 和推理调度，带准备阶段和键盘控制 |

`andy_rl_sar_reference/` 存放第二个脚本使用的策略、部署配置和模型资源。
两个脚本的时序、历史观测处理和力矩限制不完全相同，不应把它们当成等价入口。
`demo` 尚未完整适配 PACE 动力学，目录整理不改变这些仿真行为。

```bash
python scripts/sim2sim/andy_height_mujoco_demo.py --help
python scripts/sim2sim/andy_height_mujoco_rl_sar.py --help
```

Andy 的 PACE 训练文件统一位于：
`source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/andy/pace/`。

## 从 play/checkpoint 自动部署到 MuJoCo

```bash
conda activate isaaclab-pace
python scripts/sim2sim/andy_height_mujoco_rl_sar.py \
  --task DDT-Height-Flat-Andy-Pace-Robust-Play-v0 \
  --num_envs 1 \
  --keyboard \
  --checkpoint /home/htw/ddt_lab/logs/np3o/andy_height_pace_robust_stationary/2026-09-18_18-13-51
```

- `--checkpoint` 接受训练目录（按数字选择最大的 `model_N.pt`）或具体 checkpoint 文件。
- 新版 `scripts/np3o/play.py` 导出后写入 `exported/export_manifest.json`，记录 checkpoint 和 ONNX 的 SHA256。
- 如果导出记录与选择的 checkpoint/task/ONNX 不匹配，脚本先调用同一 Python 环境的 play，使用 `--export_policy --headless --num_envs 1` 导出。
  老导出没有来源记录时，会重新导出一次；需先激活支持 Isaac Lab 的 conda 环境。
- 检查 ONNX 为 `[1,27] + [1,10,27] -> [1,6]` 后，原子替换
  `andy_rl_sar_reference/policy/andymini/robot_lab/policy.onnx`，再启动 MuJoCo。
- 原训练目录和导出文件保留；部署来源写在目标旁的 `policy_source.json`。
- `--policy` 可直接指定 ONNX，不复制，不触发导出；不能与 `--checkpoint` 同用。
- `--num_envs` 仅兼容 play 命令格式。MuJoCo 始终是一台机器人；键盘控制在有窗口时可用。
- 此功能同步策略文件，不自动将训练动力学写入 rl_sar 的部署 YAML 或 MuJoCo 模型。
