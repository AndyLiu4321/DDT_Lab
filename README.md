# DDT Lab：Andy 与 PACE 使用说明

本项目使用 Isaac Lab 和 NP3O 训练 Andy（AndyMini）轮足机器人。
PACE 是 Andy 的执行器辨识参数版本，可用于固定参数训练和鲁棒续训。

## 1. 环境准备

使用已安装 Isaac Sim 5.1、Isaac Lab 2.3.0 的 Python 3.11 环境。本机示例：

```bash
conda activate isaaclab-pace
cd /home/htw/ddt_lab
python -m pip install -e source/ddt_lab
```

Andy 训练需要 URDF 和配套 mesh，默认模型路径为：

```text
ddt_ros2_control/urdfs/andymini/urdf/andymini.urdf
```

模型需自行准备；仓库未跟踪此目录中的 Andy URDF。使用其他位置时，修改
`source/ddt_lab/ddt_lab/assets/ddt_robot.py` 中的 `DDT_MODEL_DIR`。
PACE 执行器代码和候选参数已放在本项目中，训练无需额外安装 PACE 包。

检查任务注册：

```bash
python scripts/list_envs.py
```

## 2. 选择任务

### Andy 基础任务

| 用途 | 训练任务 ID | 日志目录（相对 `logs/np3o/`） |
| --- | --- | --- |
| 平地运动 | `DDT-Velocity-Flat-Andy-v0` | `andy_flat` |
| 粗糙地形运动 | `DDT-Velocity-Rough-Andy-v0` | `andy_rough` |
| 平地恢复 | `DDT-Recovery-Flat-Andy-v0` | `andy_flat` |
| 粗糙地形恢复 | `DDT-Recovery-Rough-Andy-v0` | `andy_rough` |
| 跳跃 | `DDT-jump-Flat-Andy-v0` | `andy_jump` |
| 高度控制 | `DDT-Height-Flat-Andy-v0` | `andy_height` |

### Andy PACE 任务

| 用途 | 训练任务 ID | 日志目录（相对 `logs/np3o/`） |
| --- | --- | --- |
| 固定辨识参数的高度控制基线 | `DDT-Height-Flat-Andy-Pace-v0` | `andy_height_pace` |
| 高度控制鲁棒续训 | `DDT-Height-Flat-Andy-Pace-Robust-v0` | `andy_height_pace_robust` |
| 固定 0.35 m 高度运动 | `DDT-Velocity-Flat-Andy-Pace-Fixed035-v0` | `andy_pace_fixed035` |

回放时将任务末尾的 `-v0` 改成 `-Play-v0`，例如
`DDT-Height-Flat-Andy-Pace-Robust-Play-v0`。任务名区分大小写，跳跃任务使用小写 `jump`。

## 3. Andy 训练与续训

以高度控制为例；其他任务替换 `--task` 即可：

```bash
python scripts/np3o/train.py \
  --task DDT-Height-Flat-Andy-v0 \
  --num_envs 4096 \
  --max_iterations 20000 \
  --headless
```

日志和模型保存在 `logs/np3o/<实验名>/<时间戳>/`，模型文件为 `model_N.pt`。
显存不足时减小 `--num_envs`；需要训练窗口时去掉 `--headless`。

从已有模型续训（将下方路径替换为真实 checkpoint 的绝对路径）：

```bash
python scripts/np3o/train.py \
  --task DDT-Height-Flat-Andy-v0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --resume \
  --checkpoint /absolute/path/to/model_20000.pt \
  --experiment_name andy_height_resume \
  --headless
```

`--experiment_name` 可指定新的日志目录，也适合区分共用默认目录的运动与恢复任务。

## 4. PACE 训练流程

先从头训练固定参数基线：

```bash
python scripts/np3o/train.py \
  --task DDT-Height-Flat-Andy-Pace-v0 \
  --num_envs 4096 \
  --max_iterations 20000 \
  --headless
```

基线稳定后，从该次训练的 checkpoint 进入鲁棒阶段：

```bash
python scripts/np3o/train.py \
  --task DDT-Height-Flat-Andy-Pace-Robust-v0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --resume \
  --checkpoint /absolute/path/to/andy_height_pace/run/model_20000.pt \
  --headless
```

Robust 加入动力学随机化、观测噪声和推扰课程。回放任务也保留随机化，
短时间回放不足以验证最高档推扰效果。固定 0.35 m 任务可使用同一训练入口，
将 `--task` 替换为 `DDT-Velocity-Flat-Andy-Pace-Fixed035-v0`。

候选参数位于 `source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/andy/pace/data/200hz_candidate.json`。
更换候选参数后应建立独立训练基线，避免混用不同动力学版本的模型。
执行器配置和参数细节见 [PACE 说明](source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/andy/pace/README.md)。

## 5. Isaac Lab 回放与策略导出

指定与训练相匹配的 Play 任务及模型文件：

```bash
python scripts/np3o/play.py \
  --task DDT-Height-Flat-Andy-Pace-Robust-Play-v0 \
  --checkpoint /absolute/path/to/model_5000.pt \
  --num_envs 1 \
  --keyboard
```

省略 `--checkpoint` 时，脚本按任务的实验目录自动查找模型。
如果训练使用过 `--experiment_name`，建议显式指定模型路径。
使用键盘控制时，按启动后的控制提示操作。

仅导出 TorchScript 和 ONNX，不进行回放：

```bash
python scripts/np3o/play.py \
  --task DDT-Height-Flat-Andy-Pace-Robust-Play-v0 \
  --checkpoint /absolute/path/to/model_5000.pt \
  --num_envs 1 \
  --export_policy \
  --headless
```

默认输出到模型所在目录的 `exported/`，可用 `--export_dir` 指定其他目录。

## 6. MuJoCo sim2sim

在同一环境中准备 MuJoCo 推理依赖：

```bash
python -m pip install mujoco onnxruntime pyyaml
```

使用 rl_sar 入口加载训练模型：

```bash
python scripts/sim2sim/andy_height_mujoco_rl_sar.py \
  --task DDT-Height-Flat-Andy-Pace-Robust-Play-v0 \
  --checkpoint /absolute/path/to/model_5000.pt \
  --num_envs 1 \
  --keyboard
```

这里的 `--checkpoint` 支持模型文件或训练目录；目录模式选择编号最大的 `model_N.pt`。
需要时脚本自动调用 Isaac Lab 导出策略，并更新
`scripts/sim2sim/andy_rl_sar_reference/policy/andymini/robot_lab/` 下的部署策略及来源记录。
因此自动导出需要在支持 Isaac Lab 的环境中运行。

也可用 `--policy /absolute/path/to/policy.onnx` 直接加载 ONNX，与 `--checkpoint` 二选一。
MuJoCo 始终运行一台机器人。高度任务窗口按键：W/S 前后运动，A/D 调整高度，
Q/E 转向，空格清零速度命令。

部署使用的 YAML、模型和控制时序需与目标训练配置核对；同步策略文件不会自动同步全部动力学配置。
更多入口说明见 [sim2sim 使用说明](scripts/sim2sim/README.md)。

## 7. 查看训练进度

```bash
tensorboard --logdir logs/np3o
```

需要修改任务参数时，Andy 配置位于
`source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/andy/`，
PACE 配置位于该目录下的 `pace/`。
