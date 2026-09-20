# PACE Robust 零指令续训（2026-09-18）

任务保持 DDT-Height-Flat-Andy-Pace-Robust-v0，观测/动作结构不变。

改动：零速度/零偏航指令采样比例20%→40%；机身平移与偏航惩罚不再随高度误差衰减。
新增 stationary_translation 权重-2（速度死区.01m/s），stationary_yaw 权重-.5（死区.02rad/s）。
新增 stationary_tracking 权重1：exp(-v_xy²/.05²-wz²/.1²)，只在双零指令附近生效。
腿速度惩罚独立为-.05，仍允许高度变化期间运动。action_rate_l2从-.1改为-.15。
原有增益/摩擦/惯量/质量/质心/观测噪声随机化与渐进推扰保留。
没有改变电机辨识参数、策略接口、历史定义或添加输出滤波。

代码集中在 `source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/andy/pace/`：
`stationary_mdp.py`、`robust_env_cfg.py`；历史备份保存在该目录的 `archive/`。

## 启动与日志

已用64环境2迭代完成冒烟验证，然后在物理GPU1启动4096环境5000迭代。
从旧鲁棒性训练已保存的model_3000.pt加载权重和优化器（并非仅指定--resume自动找最近实验）：

```bash
cd /home/htw/ddt_lab
bash scripts/np3o/train_andy_stationary.sh
# 实际等效命令（激活isaaclab-pace之后）：
CUDA_VISIBLE_DEVICES=1 python scripts/np3o/train.py \
  --task DDT-Height-Flat-Andy-Pace-Robust-v0 \
  --num_envs 4096 --max_iterations 5000 --resume \
  --checkpoint /home/htw/ddt_lab/logs/np3o/andy_height_pace_robust/2026-09-18_15-59-30/model_3000.pt \
  --experiment_name andy_height_pace_robust_stationary --device cuda:0 --headless
tail -f logs/np3o/launch/andy_stationary_train.log
```

注意不要在正式训练运行期间重复执行启动命令。旧GPU0训练未停止，可作对照。
日志/模型位于logs/np3o/andy_height_pace_robust_stationary/<时间>/。
runner加载权重后本次迭代计数从0开始，因此5000是本次新增优化迭代。

监控 Metrics 中 stationary_speed_mps、stationary_yaw_radps、stationary_fraction_ok
（速度<.03m/s且偏航<.05rad/s）、stationary_seconds。
这些是每环境条件平均值；没有零指令样本的环境为0，汇总均值可能被稀释。
最终应单独固定双零指令、固定高度、无随机推动、确定性actor评估，统计每高度30秒的
速度、累计位移、偏航、姿态、力矩饱和和跌倒率，并与旧checkpoint同种子对比。
训练时的动作探索、重置初始速度和推扰会影响训练日志，不能把训练均值当作静止回放结论。
高度跟踪和存活率也需同时检查，防止静止奖励妨碍平衡或高度变化。

PACE专用sim2sim执行器对齐仍是独立待办；原普通MuJoCo脚本未复现PACE偏置/惯量/摩擦等，
因此本次续训不保证解决由部署模型不一致造成的振荡。
