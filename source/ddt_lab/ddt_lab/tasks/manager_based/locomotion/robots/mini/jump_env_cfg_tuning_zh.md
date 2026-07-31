# Mini Jump 训练配置与调参指南

本文档对应 [`jump_env_cfg.py`](./jump_env_cfg.py)，任务 ID 为
`DDT-jump-Flat-Mini-v0`。目标不只是“跳高”，而是让同一个策略学会：

1. 从随机摔倒姿态恢复并站稳；
2. 识别 `jump_cmd=0/1`；
3. 按 PREP → TAKEOFF → FLIGHT → LAND 完成蓄力、起跳、腾空和稳定落地；
4. 保留对 `[vx, vy, wz]` 速度命令的响应能力。

## 1. 配置继承关系

生效配置不只来自 `jump_env_cfg.py`：

```text
MiniRoughEnvCfg
  └─ MiniFlatEnvCfg
       └─ MiniJumpFlatEnvCfg
            └─ MiniJumpFlatEnvCfg_PLAY
```

- `rough_env_cfg.py`：定义机器人、观测、动作、奖励项原始参数、随机化和仿真步长。
- `flat_env_cfg.py`：切换到平地并覆盖一部分站稳/恢复奖励。
- `jump_env_cfg.py`：新增 `jump_cmd` 状态机、跳跃观测和分阶段奖励。
- `mdp/commands.py`：`JumpCommand` 的真实状态转移与计时逻辑。
- `mdp/rewards.py`：每个奖励的返回值、门控条件和数值范围。
- `agents/np3o_cfg.py`：Mini Jump 的迭代数；基础 NP3O 超参在
  `tasks/manager_based/locomotion/agents/np3o_cfg.py`。

每次训练会把实际生效配置保存到：

```text
logs/np3o/mini_jump/<run>/params/env.yaml
logs/np3o/mini_jump/<run>/params/agent.yaml
```

比较两次实验时，以各自的 `params/env.yaml` 和 `params/agent.yaml` 为准，
不要只看当前 Python 源码。

## 2. 时间参数怎样换算

父类中：

```python
self.sim.dt = 0.005
self.decimation = 4
```

所以策略控制周期为：

```text
step_dt = 0.005 × 4 = 0.02 s
50 控制步 = 1 s
```

当前关键时间是：

| 参数 | 当前值 | 实际含义 |
|---|---:|---|
| `episode_length_s` | 15.0 s | 单回合最多 750 个控制步 |
| `trigger_step_range` | `(125, 126)` | 固定在第 125 步，约 2.5 s 进入 TAKEOFF |
| `warmup_iterations` | 500 | 前 500 个迭代不触发跳跃 |
| `steps_per_iteration` | 24 | 必须等于 NP3O `num_steps_per_env` |
| `recovery_resample_s` | 3.0 s | `jump_cmd=0` 的 RECOVERY 段重采样周期 |
| `flight_reward_window_s` | 0.8 s | 进入 FLIGHT 后高度奖励的有效窗口 |
| `s1_timeout_s` | 5.0 s | TAKEOFF 内一直未离地时的超时，继承 `JumpCommandCfg` 默认值 |
| `s3_timeout_s` | 3.0 s | LAND 阶段后重新采样，继承 `JumpCommandCfg` 默认值 |

`torch.randint` 不包含上界，因此 `(125, 126)` 不是 125–126 随机，而是永远取 125。
若需要 2–3 s 随机蓄力，可设为 `(100, 151)`。

## 3. 当前奖励结构

权重不等于该项对总 reward 的实际贡献：各 reward 函数的原始返回范围不同，
生效的阶段和持续时间也不同。例如 `jump_flight_height` 的原始值可高于 1，
而 LAND 的两个指数奖励上限约为 1。判断谁在主导训练时，应比较 TensorBoard 中的
`Episode/Episode_Reward/<name>` 实际曲线，不要只比较权重数字。

### 恢复和站稳项

| 奖励项 | 当前权重 | 作用 | 权重过大的风险 |
|---|---:|---|---|
| `base_contact_raw` | -2.0 | base_link 接地扣分，不因倒置而失效 | 还没学会起身就持续得大额负奖励 |
| `upright_progress` | +2.0 | 提供从完全倒置到直立的线性梯度 | 站稳收益压过跳跃 |
| `inverted_ang_vel_bonus` | +0.1 | 倒置时鼓励产生自救转动 | 反复翻滚/甩动刷分 |
| `base_height_l2` | -10.0 | 直立时约束站立高度为 0.35 m | 起跳初期被过度拉回站立高度 |
| `flat_orientation_l2` | -5.0 | 惩罚过大 roll/pitch | 限制起跳和空中姿态调整 |
| `upward` | +1.0 | 给予通用直立奖励 | 低头站稳比跳跃更容易得分 |
| `track_lin_vel_xy_exp` | +0.6 | 保留 `vx/vy` 跟踪 | 策略优先跑动而不跳 |
| `track_ang_vel_z_exp` | +0.3 | 保留 `wz` 跟踪 | 转向任务干扰起跳和落地 |

`lin_vel_z_l2` 和 `ang_vel_xy_l2` 在跳跃配置中都为 0，因为它们会分别惩罚
起跳所需的向上速度和空中姿态调整。

### 分阶段跳跃项

| 阶段 | 奖励项 | 当前权重 | 生效条件 |
|---|---|---:|---|
| PREP | `jump_before_setting` | +2.0 | `jump_cmd=1`、触发前、至少一足接地 |
| TAKEOFF/FLIGHT | `lin_vel_z_jump` | +18.0 | 状态处于 TAKEOFF 或 FLIGHT，只奖励正 z 速度 |
| FLIGHT | `jump_flight_height` | +28.0 | 已离地、处于 0.8 s 窗口且 COM 高于 0.42 m |
| LAND | `jump_land_stability` | +5.0 | 落地后奖励低 roll/pitch 角速度 |
| LAND | `jump_land_orientation` | +3.0 | 落地后奖励直立姿态 |

`jump_land_stability` 的函数在有 `jump_cmd` 状态机时返回稳定度指数奖励，
所以当前应用正权重，不要因为函数内旧的英文说明而改成负权重。

## 4. 必须联动修改的参数

### 目标高度

`target_height` 在两处出现，必须保持一致：

```python
self.commands.jump_cmd = mdp.JumpCommandCfg(
    target_height=0.8,
    # ...
)
self.rewards.jump_flight_height.params["target_height"] = 0.8
```

如果只改 command 中的值，critic 的 `max_height` 归一化标准会变，但 reward 仍然按旧目标计算；
如果只改 reward，则观测归一化和奖励目标不一致。

### warmup 换算

`steps_per_iteration=24` 必须与
`tasks/manager_based/locomotion/agents/np3o_cfg.py` 中的
`runner.num_steps_per_env=24` 一致。如果改了 runner，这里也要同步改。

### 观测维度

Mini Jump 相比 Mini Flat 多了 actor 的 `jump_cmd` 和 critic 的
`jump_cmd + jump_state`。当前 checkpoint 期望的 policy/critic 维度是 `28/49`。
修改观测项后不要直接加载旧 checkpoint，除非已确认维度和顺序都完全相同。

## 5. 按现象调参

下表中的数字是建议的“一次试验变化量”，不是保证最优的固定值。
每次只改一组相关参数，否则无法判断是哪个改动起作用。

| 训练现象 | 优先检查 | 第一步建议 | 不要立即做 |
|---|---|---|---|
| 500 迭代前连站稳/起身都没学会 | reset 姿态是否过难，`upright_progress`、`base_contact_raw` | `warmup_iterations: 500 → 800`，或将 roll/pitch reset 范围暂时缩小 20%–30% | 提高跳跃奖励 |
| 只会翻滚、甩动，不愿站稳 | `inverted_ang_vel_bonus` 刷分 | `0.1 → 0.05`，仍刷分则改为 0 | 继续提高角速度奖励 |
| 站得很稳，但 `jump_cmd=1` 也不跳 | jump 样本是否太少，TAKEOFF 是否有向上速度 | `jump_probability: 0.2 → 0.25/0.3`；若样本已足够，`lin_vel_z_jump: 18 → 22` | 同时大幅提高速度奖励和高度奖励 |
| 会伸腿或小幅弹动，但不是真正腾空 | `min_height`、足接触判定、`max_height` | `min_height: 0.42 → 0.45/0.46`；确认两个 `.*_leg_4` 能正确进入 FLIGHT | 盲目把 `target_height` 提高到不可达值 |
| 能离地但高度不足 | `lin_vel_z_jump`、`jump_flight_height`、蓄力姿态 | 先将 `lin_vel_z_jump: 18 → 22`；若起跳速度已明显上升再将 `jump_flight_height: 28 → 32` | 一次将两者翻倍 |
| 没有明显下蹲蓄力 | `crouch_height=0.25`、`sigma=0.04`、PREP 时长 | `jump_before_setting: 2 → 3`，或把 `sigma: 0.04 → 0.05` 放宽容差 | 将 `crouch_height` 降到超出关节可达范围 |
| 下蹲后长时间不起跳 | PREP 触发时间 | `trigger_step_range: (125,126) → (75,101)`，即约 1.5–2.0 s | 增加下蹲奖励 |
| 跳得高但落地翻倒 | LAND 奖励、空中角速度 | `jump_land_stability: 5 → 7`，再观察；若姿态倾斜则 `jump_land_orientation: 3 → 4/5` | 立即恢复强 `ang_vel_xy_l2` 全阶段惩罚 |
| 跳跃学会后又只站立不跳 | 奖励占比和 `jump_cmd` 样本率 | `jump_probability: 0.2 → 0.3`，或 `upward: 1.0 → 0.7` | 删掉站稳奖励，这会丢失 `jump_cmd=0` 行为 |
| 只会跳，`jump_cmd=0` 也乱跳 | `jump_probability`、观测项、分阶段门控 | `jump_probability: 0.2` 保持或降到 0.15；确认 actor 的 `jump_cmd` 观测未被删除 | 继续提高跳跃样本比例 |
| 跳跃正常，但 `vx/wz` 跟踪差 | 速度命令范围和跟踪奖励 | 完成跳跃后再将 `track_lin: 0.6 → 0.8`、`track_ang: 0.3 → 0.4` | 在还未学会跳时扩大到高速命令 |

### 修改下蹲目标和容差

`crouch_height` 和 `sigma` 的默认值在 `rough_env_cfg.py` 中。要在 Jump 任务中覆盖，
在 `jump_env_cfg.py` 的 PREP 段加：

```python
self.rewards.jump_before_setting.params["crouch_height"] = 0.25
self.rewards.jump_before_setting.params["sigma"] = 0.05
```

- `crouch_height` 决定期望的 COM 下蹲高度。
- `sigma` 越大，容许得分的高度区间越宽，奖励更密集；越小则目标更精确，但更难学。

## 6. 建议的调参顺序

不建议开始就同时调所有奖励。按以下顺序更容易定位问题。

### 阶段 A：先确认恢复和站稳

在 0–500 迭代观察：

- 从侧倒、仰躺、俯卧能否逐渐恢复；
- base 接地时是起身，还是用翻滚/甩动刷分；
- `Train/mean_episode_length` 是否接近 750 控制步；
- `Policy/mean_noise_std`、`Loss/value_function`、`Data/obs_max/min` 是否突然爆炸。

由于 `base_contact` termination 已关闭，回合长并不能单独证明已学会起身，
必须结合视频/回放和各奖励曲线。

### 阶段 B：再确认是否真正离地

在 warmup 结束后优先看：

- `jump_cmd` 是否在 warmup 后出现稳定的非零样本；
- `has_jumped` 是否从 0 开始上升；
- `max_height` 是否明显高于正常站立高度 0.35 m；
- `lin_vel_z_jump` 和 `jump_flight_height` 是否同时出现非零值。

只有 `lin_vel_z_jump` 上升而 `jump_flight_height` 长期为 0，通常表示策略在向上伸腿，
但没有被接触状态机识别为真正 FLIGHT。

TensorBoard 中的 `jump_cmd` 是按控制步累积的时间平均，而 jump/no-jump 段的持续时间不同，
因此该曲线不会严格等于 `jump_probability=0.2`。它适合确认指令是否生效和比较实验趋势，
不适合当作精确的分段采样率。

### 阶段 C：最后调落地和速度跟踪

当 `has_jumped` 和 `max_height` 已稳定后，再提高 LAND 奖励以及 `vx/wz` 跟踪权重。
如果在尚未学会离地时就强化落地和位移，策略很容易选择“不跳最安全”。

## 7. 训练、回放和对比

### 开始新训练

```bash
python scripts/np3o/train.py \
  --task DDT-jump-Flat-Mini-v0 \
  --num_envs 4096 \
  --max_iterations 20000 \
  --headless
```

调参时建议为每次实验单独设置名称，避免日志混在一起：

```bash
python scripts/np3o/train.py \
  --task DDT-jump-Flat-Mini-v0 \
  --num_envs 4096 \
  --max_iterations 20000 \
  --experiment_name mini_jump_height_v2 \
  --headless
```

### 继续训练

```bash
python scripts/np3o/train.py \
  --task DDT-jump-Flat-Mini-v0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --resume \
  --load_checkpoint /absolute/path/to/model_XXXX.pt \
  --headless
```

`--max_iterations 5000` 表示本次进程再学习 5000 次，不是训练到绝对迭代号 5000。
观测和动作维度不变时，修改 reward 权重、command 概率/时间或 reset 范围可以尝试续训；
但变化很大时最好同时启动一个新 run 作为对照。改变观测维度/顺序、动作维度或机器人关节顺序后应重新训练。

### 回放 checkpoint

```bash
python scripts/np3o/play.py \
  --task DDT-jump-Flat-Mini-Play-v0 \
  --num_envs 50 \
  --keyboard \
  --checkpoint /absolute/path/to/model_XXXX.pt
```

键盘模式下：

- `W/S`：前进/后退；
- `A/D`：左/右平移；
- `Q/E`：左/右转向；
- `R`：切换跳跃指令；
- `Space`：清空速度和跳跃指令。

### TensorBoard

```bash
tensorboard --logdir logs/np3o --port 6006
```

至少同时观察：

- 跳跃 command metrics：`Episode/Metrics/jump_cmd/jump_cmd`、
  `Episode/Metrics/jump_cmd/jump_stage`、`Episode/Metrics/jump_cmd/has_jumped`、
  `Episode/Metrics/jump_cmd/max_height`；
- 分项 reward：`Episode/Episode_Reward/jump_before_setting`、
  `Episode/Episode_Reward/lin_vel_z_jump`、`Episode/Episode_Reward/jump_flight_height`、
  `Episode/Episode_Reward/jump_land_stability`、`Episode/Episode_Reward/jump_land_orientation`；
- 恢复项：`Episode/Episode_Reward/base_contact_raw`、
  `Episode/Episode_Reward/upright_progress`、`Episode/Episode_Reward/inverted_ang_vel_bonus`；
- 稳定性：`Loss/value_function`、`Loss/learning_rate`、`Policy/mean_noise_std`；
- 观测异常：`Data/obs_max`、`Data/obs_min`；
- 整体表现：`Train/mean_reward`、`Train/mean_episode_length`。

不要只看 `Train/mean_reward`：调整 reward 权重后，总 reward 的数值标尺本身就变了。
应优先比较 `has_jumped`、`max_height`、落地姿态和回放行为。

## 8. 一组保守的下一轮实验方案

如果当前策略“能站稳，但跳跃频率低/跳不起来”，可先只试下面三个改动：

```python
jump_probability=0.3

self.rewards.lin_vel_z_jump.weight = 22.0
self.rewards.jump_flight_height.weight = 28.0  # 先不动高度奖励
```

先跑到 warmup 后 500–1000 个迭代，检查 `has_jumped` 和 `max_height`：

- `has_jumped` 上升且 `max_height` 上升：保留该改动，再单独调 LAND 奖励。
- `lin_vel_z_jump` 上升但 `has_jumped` 仍为 0：检查接触 sensor 和 `min_height`，不要继续堆奖励。
- `jump_cmd` 已有足够样本但向上速度仍为 0：检查蓄力姿态、动作范围和执行器是否有足够起跳能力。
- 学会翻滚刷分：将 `inverted_ang_vel_bonus` 从 0.1 降到 0.05 或 0。

对于其他症状，不应盲目套用这组值，而应按第 5 节的现象表选择对应参数。

# sim2sim
```bash
python scripts/sim2sim/mini_jump_mujoco_demo.py \
    --policy logs/np3o/mini_jump/2026-07-14_16-41-47/exported/policy.pt \
    --jump-cmd 0.3 --vx 0.5 --vy 0 --wz 0
    --disable-scene-obstacles
```