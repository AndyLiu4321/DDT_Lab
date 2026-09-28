
### sim2sim
```bash
# 只验证 MuJoCo + PD 链路
conda activate isaaclab
python scripts/sim2sim/tita_mujoco_demo.py --headless --duration 2

# 用 NP3O command-gated policy 做 sim2sim，平地前进
python scripts/sim2sim/tita_mujoco_demo.py \
  --policy logs/np3o/tita_command_gated_flat/2026-07-09_10-13-00/exported/policy.pt \
  --profile command_gated \
  --vx 0 \
  --disable-scene-obstacles
```
```bash
python scripts/sim2sim/tita_mujoco_demo.py \
  --policy logs/np3o/tita_rough/2026-06-11_17-43-15/exported/policy.pt \
  --profile velocity \
  --vx -1 \
  --wz 1 \
  --disable-scene-obstacles
python scripts/sim2sim/tita_mujoco_demo.py \
  --profile jump \
  --duration 500 \
  --policy /home/htw/ddt_lab/logs/np3o/tita_jump/2026-07-16_17-31-02/exported/policy.pt

  
python scripts/sim2sim/tita_mujoco_demo.py \
  --policy logs/np3o/tita_flat/2026-06-16_14-37-15/exported/policy.pt \
  --profile velocity \
  --vx -1 \
  --disable-scene-obstacles
python scripts/sim2sim/mini_jump_mujoco_demo.py \
  --vx 0 \
  --policy /home/htw/ddt_lab/logs/np3o/mini_jump/2026-07-03_11-42-49/exported/policy.pt

python scripts/sim2sim/mini_jump_mujoco_demo.py \
    --policy logs/np3o/mini_jump/2026-07-14_16-41-47/exported/policy.pt \
    --jump-cmd 0 --vx 0 --vy 0 --wz 0 --disable-scene-obstacles

# python scripts/np3o/play.py \
#   --task DDT-jump-Flat-Mini-Play-v0 \
#   --num_envs 50 \
#   --checkpoint "/home/htw/ddt_lab/logs/np3o/mini_jump/2026-07-03_11-42-49/model_5400.pt" \

python scripts/sim2sim/mini_flat_mujoco_demo.py \
  --duration 50 \
  --disable-scene-obstacles \
  --policy /home/htw/ddt_lab/logs/np3o/mini_flat/2026-07-13_10-55-51/exported/policy.pt

python scripts/sim2sim/d1_rough_mujoco_demo.py \
  --duration 50 \
  --disable-scene-obstacles \
  --vx 0.3 \
  --heading 0.0

miniconda3/envs/isaaclab/bin/python scripts/sim2sim/d1_rough_mujoco_demo.py \
  --headless \
  --duration 5 \
  --disable-scene-obstacles \
  --vx 0.3 \
  --heading 0.0

/home/htw/miniconda3/envs/isaaclab/bin/python scripts/sim2sim/d1_rough_mujoco_demo.py \
  --duration 50 \
  --disable-scene-obstacles \
  --vx 0.3 \
  --heading 0.0

/home/htw/miniconda3/envs/isaaclab/bin/python scripts/sim2sim/mini_flat_mujoco_demo.py \
  --duration 50 \
  --disable-scene-obstacles


```