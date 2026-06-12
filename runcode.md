# DDT Lab Run Code (NP3O + ROS2/MuJoCo)

## 0. 环境准备

```bash
cd /home/htw/ddt_lab
conda activate isaaclab
python -m pip install -e source/ddt_lab
```

验证任务是否注册成功：

```bash
python scripts/list_envs.py
```

---

## 1. 训练（README 推荐）

### D1 Flat

```bash
python scripts/np3o/train.py --task=DDT-Velocity-Flat-D1-v0 \
  --num_envs 4096 \
  --headless
```

### D1 Rough

```bash
python scripts/np3o/train.py --task=DDT-Velocity-Rough-D1-v0 \
  --num_envs 4096 \
  --headless
```

### Tita Flat

```bash
python scripts/np3o/train.py --task=DDT-Velocity-Flat-Tita-v0 \
  --num_envs 4096 \
  --headless
```

---

## 2. 续训（README 模板）

```bash
python scripts/np3o/train.py --task=DDT-Velocity-Flat-D1-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --load_run ".*" \
  --load_checkpoint "model_.*\\.pt"
```

---

## 3. 评估 / 回放（README）

### 自动加载最新 checkpoint

```bash
python scripts/np3o/play.py --task=DDT-Velocity-Flat-D1-Play-v0
```

### 指定 checkpoint

```bash
python scripts/np3o/play.py --task=DDT-Velocity-Flat-D1-Play-v0 \
  --checkpoint /path/to/model_5000.pt
```

### 导出 JIT + ONNX（不 rollout）

```bash
python scripts/np3o/play.py --task=DDT-Velocity-Flat-D1-Play-v0 \
  --export_policy \
  --export_dir /tmp/d1_deploy
```

---

## 4. 你的常用命令（保留）

### Tita Rough 训练（两阶段）

```bash
python scripts/np3o/train.py \
  --task=DDT-Velocity-Rough-Tita-v0 \
  --num_envs 4096 \
  --termination_stage 2 \
  --max_iterations 20000 \
  --headless 
```

```bash
python scripts/np3o/train.py \
  --task=DDT-Velocity-Rough-Tita-v0 \
  --num_envs 4096 \
  --termination_stage 1 \
  --resume \
  --load_run "2026-06-11_10-40-47" \
  --load_checkpoint "model_1000.pt" \
  --headless
```

### Tita Rough 回放

```bash
python scripts/np3o/play.py \
  --task DDT-Velocity-Rough-Tita-Play-v0 \
  --num_envs 50 \
  --checkpoint "/home/htw/ddt_lab/logs/np3o/tita_rough/2026-06-11_17-43-15/model_20000.pt" \
  --termination_stage 1 \
  --headless \
  --video
python scripts/np3o/play.py \
  --task DDT-Velocity-Rough-Tita-Play-v0 \
  --num_envs 50 \
  --checkpoint "/home/htw/ddt_lab/logs/np3o/tita_rough/2026-06-09_11-56-51/model_2000.pt" \
  
```

### D1 Rough 训练

```bash
python scripts/np3o/train.py \
  --task=DDT-Velocity-Rough-D1-v0 \
  --num_envs 4096 \
  --termination_stage 1 \
  --max_iterations 2000 \
  --headless
```

### D1 Rough 回放

```bash
python scripts/np3o/play.py \
  --task DDT-Velocity-Rough-D1-Play-v0 \
  --num_envs 50 \
  --checkpoint "/home/htw/ddt_lab/logs/np3o/d1_rough/2026-06-09_15-20-40/model_2600.pt" \
  --headless \
  --video
```

---

## 5. ROS2 + MuJoCo（ddt_ros2_control）

### 键盘控制节点

```bash
source /opt/ros/humble/setup.bash
source /home/htw/ddt_ros2_ws/install/setup.bash
ros2 run keyboard_controller keyboard_controller_node
```

### 启动 Tita Mujoco

```bash
source /opt/ros/humble/setup.bash
source /home/htw/ddt_ros2_ws/install/setup.bash
export MUJOCO_DIR=/home/htw/.local/mujoco-3.3.0
export LD_LIBRARY_PATH=$MUJOCO_DIR/lib:$LD_LIBRARY_PATH
ros2 launch rl_controller sim_mujoco.launch.py robot:=tita
```

### 启动 D1 Mujoco

```bash
source /opt/ros/humble/setup.bash
source /home/htw/ddt_ros2_ws/install/setup.bash
export MUJOCO_DIR=/home/htw/.local/mujoco-3.3.0
export LD_LIBRARY_PATH=$MUJOCO_DIR/lib:$LD_LIBRARY_PATH
ros2 launch rl_controller sim_mujoco.launch.py robot:=d1
```

---

## 6. 监控训练

```bash
tensorboard --logdir logs/np3o
```

## 7. 环境冒烟测试

```bash
python scripts/zero_agent.py --task=DDT-Velocity-Flat-D1-v0
python scripts/random_agent.py --task=DDT-Velocity-Flat-Tita-v0
```