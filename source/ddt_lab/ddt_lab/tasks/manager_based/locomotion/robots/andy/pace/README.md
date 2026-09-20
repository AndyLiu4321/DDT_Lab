# Andy PACE 文件索引

```text
pace/
├── __init__.py          # 任务注册、历史模块导入兼容
├── agents.py            # NP3O runner 配置
├── actuator.py          # PACE 电机实现
├── actuator_cfg.py      # 执行器配置
├── height_env_cfg.py    # 固定参数 Height 配置
├── robust_env_cfg.py    # 随机化及静止微调配置
├── robust_mdp.py        # 推扰课程与高度统计
├── stationary_mdp.py    # 静止命令及奖励
├── data/200hz_candidate.json
├── archive/             # 历史备份，不参与运行
└── README.md
```

原任务 ID、训练命令保持不变。新代码使用 `andy.pace.*` 导入；
历史 `andy.pace_height_env_cfg` 等模块路径由兼容别名解析到同一份代码，
旧日志和导出配置无需批量改写。原 `andy.agents.np3o_cfg` 的 PACE runner 入口也保留兼容。
当前 Robust 配置包含静止微调修改，详见项目根目录 `ANDY_STATIONARY_TRAINING.md`。

# Andy Height：PACE 候选执行器

新增任务 `DDT-Height-Flat-Andy-Pace-v0` 和 `DDT-Height-Flat-Andy-Pace-Play-v0`。
原 Height 任务继续使用原配置。入口为 `height_env_cfg.py`。

参数来自 `26_09_17_17-19-36` 第 110 代最优已评估个体，保存在
`data/200hz_candidate.json`。不是 `mean_110.pt` 中的 CMA 分布均值。
`actuator.py` / `actuator_cfg.py` 来自本机 PACE 项目，保留原许可证，
仅将配置中的模块导入改成本地相对路径；训练不依赖 PACE 包。

## 坐标和控制

- 四个腿关节和两个驱动轮均使用 PACE 显式 DCMotor，辅助轮配置保留。
- PD：腿 Kp=40/Kd=1，轮 Kp=0/Kd=0.5；力矩上限 11 N m，速度上限 12.57 rad/s。
- 同时设置 armature、viscous_friction、friction、dynamic_friction、encoder_bias。
- 此候选实际延迟为 0 步；不允许直接替换成非零延迟候选。
- PACE 实现的符号为 `q_encoder = q_true - bias`。
- 物理初始位置和 default_joint_pos 使用 `q_nominal + bias`。
  原相对位置观测 `q_true - default_joint_pos` 因而等于 `q_encoder - q_nominal`，不要再次减 bias。
- 位置动作关闭 use_default_offset，显式使用原编码器 nominal（0 或 0.66）。
  动作尺度沿用 Height（腿 1 为 0.15，腿 2 为 0.66，轮速为 10）。
- 物理位置限位转换成编码器目标限位 `limit_true - bias`。
  几何奖励和物理位置成本仍读取真实关节位置；没有将整个 articulation 数据改成编码器值。
- 初始关节事件改为默认物理位置加零偏移、速度为零。根状态随机初始化仍沿用 Height。

## 时间与随机化

物理步长 1 ms，训练 decimation=20，保持原来的 20 ms/50 Hz 策略周期。
这按此前 200 Hz 拟合命令选择；历史 config.pt 没有保存物理步长。
实机扫频回放则是 decimation=5（200 Hz），不使用训练的策略动作尺度。

当前配置用于固定参数基线：关闭质量、惯量、质心、PD 增益、接触材质随机化，
关闭持续外力、推扰以及策略/critic 观测噪声。高度命令、奖励、终止规则沿用 Height。
地面接触参数、基座自由运动和高度性能不在此次固定基座 PACE 辨识的验证范围内。
之后可在此基线上另加域随机化；先保留一份固定参数对照。

## 运行

```bash
cd /home/htw/ddt_lab
conda activate isaaclab-pace
python scripts/np3o/train.py \
  --task DDT-Height-Flat-Andy-Pace-v0 \
  --num_envs 256 --max_iterations 20000 --headless
```

日志目录：`logs/np3o/andy_height_pace/`。建议新建训练，不默认续训旧动力学的 checkpoint。
回放：

```bash
python scripts/np3o/play.py \
  --task DDT-Height-Flat-Andy-Pace-Play-v0 \
  --num_envs 1 --keyboard \
  --checkpoint /home/htw/ddt_lab/logs/np3o/andy_height_pace/2026-09-20_11-33-24/model_2100.pt
```

固定基座回放（在 PACE 项目中运行，需要本机两个包均已安装）：

```bash
cd /home/htw/pace-sim2real
python scripts/pace/replay_andy_candidate.py --headless
```

结果保存到 `data/andy_real/fit_comparison/lab_replay.pt`。
回放使用 Lab 迁移后的电机、PACE 固定基座场景、实机原目标和原初始状态。
这验证的是执行器迁移，不是高度策略成功率。

## 已完成的验证（2026-09-18）

- 配置检查、两环境五步仿真通过，驱动轮实际实例为 PaceDCMotor。
- 使用迁移后的 Lab 电机完整回放 6000 帧 200 Hz 实机目标：
  加权误差 `0.0005746159586`，原 TensorBoard 最优误差 `0.0005746154347`。
  六关节 RMSE 分别为 `[0.01003571, 0.00926990, 0.04390029, 0.01337798, 0.01266373, 0.05418188]`。
  与原候选轨迹误差在数值精度内一致，支持此处 1 ms/decimation=5 的回放配置。
- 这组参数包含接近上界的腿 1 编码器偏差；复现成功不等于物理参数已独立验证，
  也不等于高度策略已经收敛或可以直接部署实机。
- 16 环境、1 轮 NP3O 训练通过，共 384 步采样；此检查仅验证训练链路。

## 适度随机化与抗扰续训（2026-09-18）

新增 `DDT-Height-Flat-Andy-Pace-Robust-v0`，配置为 `robust_env_cfg.py`。
固定版与正在运行的训练进程不受影响，编码器偏差仍固定在 PACE 候选值。
以下范围是保守的训练起点，并非由实机测得的不确定性区间：

| 项目 | 范围 |
| --- | --- |
| PD 增益 | 每次 reset，相对 PACE 基准 ±10% |
| 电机关节 armature、摩擦系数及黏性摩擦 | startup，相对 PACE 基准 ±10% |
| 基座质量 | startup，±5% |
| 基座质心 | startup，XYZ 各 ±3 mm |
| 地面接触 | 静摩擦 0.7–1.0，动摩擦 0.6–0.9，恢复系数 0–0.05；动摩擦不超过静摩擦 |
| 初始关节角/速度 | 相对物理默认姿态 ±0.03 rad，±0.05 rad/s |
| Actor 观测噪声 | 角速度 ±0.1 rad/s；重力投影 ±0.02；关节角 ±0.005 rad；关节速度 ±0.2 rad/s |

Critic 保持无噪声。刚体惯量额外随机化、持续外力和偏差随机化仍关闭。

推扰为每隔 8–12 秒向世界坐标 XY 根速度各添加一次随机增量，
不是持续外力，不改变 Z、roll、pitch 或 yaw 速度。
课程档位为每轴最大增量 `[0, 0.05, 0.10, 0.15, 0.20] m/s`。
每档至少收集 5000 个策略步（单环境 100 秒）和 1024 个已结束回合：
非失败回合比例 >=95%，且按实际回合时长归一化的高度奖励质量 >=0.9，才升一档；
非失败比例 <90% 或高度质量 <0.8，降一档；否则保持。
初始 reset 不计入统计。XY 同时取极值时合速度增量可达 sqrt(2)*每轴上限。
每个新训练进程从零推扰重新评估，不从旧 checkpoint 恢复课程状态。
TensorBoard 记录 `Curriculum/pace_push/{stage,max_delta_v,survival,height_quality}`。

高度命令保留历史 height_error 指标，并新增 height_mae_m。
历史指标按 10 秒除累计误差、在回合 reset 清零，20 秒完整回合下约为真实 MAE 的两倍。
height_mae_m 用实际更新次数归一化；短回合也不再因固定时间分母被低估。

从已稳定的固定版候选续训（原 model_5600.pt 不修改）：

```bash
cd /home/htw/ddt_lab
conda activate isaaclab-pace
python scripts/np3o/train.py \
  --task DDT-Height-Flat-Andy-Pace-Robust-v0 \
  --num_envs 4096 --max_iterations 5000 \
  --resume \
  --checkpoint /home/htw/ddt_lab/logs/np3o/andy_height_pace/2026-09-18_08-52-36/model_5600.pt \
  --headless
```

日志：`logs/np3o/andy_height_pace_robust/`。本项目当前 load 加载模型和优化器，
新阶段的训练迭代计数从零开始。原训练仍在运行时，另启 4096 环境会共享 GPU 资源；
可在原训练终端结束固定版训练后启动此阶段。

已通过 16 环境、1 轮 checkpoint 续训检查（384 步采样），无维度不匹配。
另测试了课程初始排除、升级、上限、失稳降级、低高度质量不升级以及 XY 扰动范围。
未启动正式 5000 轮训练；短检查不代表已验证随机化后的收敛和抗扰效果。
Robust-Play 任务保留随机化和噪声；短回放通常不会达到课程门槛，不能据此声称通过最大推扰测试。
