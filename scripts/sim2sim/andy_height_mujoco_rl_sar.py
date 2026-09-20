#!/usr/bin/env python3
"""Standalone AndyMini ONNX sim2sim, using the local rl_sar deployment contract.

python scripts/sim2sim/andy_height_mujoco_rl_sar.py --height .22
python scripts/sim2sim/andy_height_mujoco_rl_sar.py --headless --duration 20 --csv /tmp/andy.csv
Use --checkpoint TRAINING_DIR (or model_N.pt) to export/copy a policy automatically.
Viewer keys: W/S forward, A/D height +/- .01, Q/E yaw, SPACE zero velocity.
Time is simulation time: physics 2ms, PD 5ms, inference 20ms, no ROS/C++ needed.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
if (Path(__file__).resolve().parent / 'andy_rl_sar_reference').is_dir():
    ROOT = Path(__file__).resolve().parent / 'andy_rl_sar_reference'
JOINTS = [f"joint_{side}_leg_{i}" for side in ("left", "right") for i in (1, 2, 3)]
WHEELS = np.array([2, 5])


def select_checkpoint(path):
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = [(int(m.group(1)), p) for p in path.glob('model_*.pt')
                      if (m := re.fullmatch(r'model_(\d+)\.pt', p.name))]
        if not candidates:
            raise FileNotFoundError(f'No model_<iteration>.pt in {path}')
        return max(candidates, key=lambda item: item[0])[1]
    if not path.is_file() or path.suffix != '.pt':
        raise ValueError('--checkpoint must be a training directory or a checkpoint .pt file')
    if path.name == 'policy.pt':
        raise ValueError('Use model_<iteration>.pt, not exported TorchScript policy.pt')
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_policy(args):
    """Verify/export the selected checkpoint and atomically copy its ONNX."""
    destination = ROOT / 'policy/andymini/robot_lab/policy.onnx'
    if args.checkpoint is None:
        args.policy = args.policy or destination
        return
    checkpoint = select_checkpoint(args.checkpoint)
    exported = checkpoint.parent / 'exported'
    source = exported / 'policy.onnx'
    manifest_path = exported / 'export_manifest.json'
    fingerprint = sha256(checkpoint)

    def valid_export():
        try:
            manifest = json.loads(manifest_path.read_text())
            return (manifest['checkpoint'] == str(checkpoint)
                    and manifest['checkpoint_sha256'] == fingerprint
                    and manifest['task'] == args.task
                    and manifest['onnx_sha256'] == sha256(source))
        except (OSError, ValueError, KeyError):
            return False

    print(f'[policy] Selected checkpoint: {checkpoint}', flush=True)
    if not valid_export():
        print('[policy] Export missing or provenance does not match; running play export.', flush=True)
        # A single Isaac environment suffices to export the network. MuJoCo also
        # displays one robot; --num_envs is accepted for CLI compatibility only.
        command = [sys.executable, str(REPO_ROOT / 'scripts/np3o/play.py'),
                   '--task', args.task, '--checkpoint', str(checkpoint),
                   '--num_envs', '1', '--export_policy', '--headless']
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        if not valid_export():
            raise RuntimeError('Export did not produce a matching manifest; deployment unchanged')
    if sha256(checkpoint) != fingerprint:
        raise RuntimeError('Checkpoint changed during export; retry after it finishes saving')
    # Validate before replacing the working deployment artifact.
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(source), sess_options=options, providers=['CPUExecutionProvider'])
    io = session.get_inputs() + session.get_outputs()
    if (len(session.get_inputs()) != 2 or len(session.get_outputs()) != 1
            or [(v.name, v.shape, v.type) for v in io] != [
                ('nn_input0', [1, 27], 'tensor(float)'),
                ('nn_input1', [1, 10, 27], 'tensor(float)'),
                ('nn_output', [1, 6], 'tensor(float)')]):
        raise ValueError('Export is not an Andy 27/10x27 -> 6 policy; deployment unchanged')
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != json.loads(manifest_path.read_text())['onnx_sha256']:
        raise RuntimeError('ONNX changed during deployment; retry after export completes')
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.onnx.tmp', delete=False) as f:
        temporary = Path(f.name)
        f.write(payload)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    provenance = json.loads(manifest_path.read_text())
    provenance['source_onnx'] = str(source)
    destination.with_name('policy_source.json').write_text(json.dumps(provenance, indent=2) + '\n')
    args.policy = destination
    print(f'[policy] Copied {source} -> {destination} (original retained)', flush=True)


def rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class Simulation:
    def __init__(self, args):
        self.args = args
        with args.config.open() as f:
            self.cfg = yaml.safe_load(f)["andymini/robot_lab"]
        c = self.cfg
        self.home = np.array(c["default_dof_pos"], dtype=float)
        self.kp = np.array(c["rl_kp"], dtype=float)
        self.kd = np.array(c["rl_kd"], dtype=float)
        if args.kp is not None:
            self.kp[[0,1,3,4]] = args.kp
        if args.kd is not None:
            self.kd[[0,1,3,4]] = args.kd
        if args.wheel_kd is not None:
            self.kd[WHEELS] = args.wheel_kd
        self.limits = np.array(c["torque_limits"], dtype=float)
        if args.torque_limit is not None:
            self.limits[:] = args.torque_limit
        self.model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
        self.model.opt.timestep = args.dt
        self.data = mujoco.MjData(self.model)
        def index(kind, name):
            value = mujoco.mj_name2id(self.model, kind, name)
            if value < 0:
                raise ValueError(f"Missing MuJoCo object: {name}")
            return value
        ids = [index(mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINTS]
        self.qa = self.model.jnt_qposadr[ids]
        self.va = self.model.jnt_dofadr[ids]
        self.act = np.array([index(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS])
        self.base = index(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.site = index(mujoco.mjtObj.mjOBJ_SITE, "trunk_imu")
        # Model's own force limits remain an additional bound.
        self.limits = np.minimum(self.limits, self.model.actuator_ctrlrange[self.act, 1])
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.policy = ort.InferenceSession(str(args.policy), sess_options=options,
                                          providers=["CPUExecutionProvider"])
        for obj, name, shape in zip(self.policy.get_inputs()+self.policy.get_outputs(),
                                   ["nn_input0", "nn_input1", "nn_output"],
                                   [[1,27], [1,10,27], [1,6]]):
            if obj.name != name or obj.shape != shape or obj.type != "tensor(float)":
                raise ValueError(f"ONNX contract mismatch: {obj.name} {obj.shape} {obj.type}")
        if len(self.policy.get_inputs()) != 2 or len(self.policy.get_outputs()) != 1:
            raise ValueError("Expected exactly two inputs and one output")
        self.command = np.array([args.vx, args.height, args.wz], dtype=float)
        self.command = np.clip(self.command, c["command_lower"], c["command_upper"])
        self.action = np.zeros(6, dtype=np.float32)
        self.history = None
        self.target = self.home.copy()
        self.velocity = np.zeros(6)
        self.raw_tau = np.zeros(6)
        self.obs = np.zeros(27, dtype=np.float32)
        self.last_hist = np.zeros((10,27), dtype=np.float32)
        self.tilts, self.heights, self.actions, self.torques = [], [], [], []
        self.first_fall = None
        mujoco.mj_forward(self.model, self.data)
        self.initial_q = self.data.qpos[self.qa].copy()
        print("policy sha256:", hashlib.sha256(args.policy.read_bytes()).hexdigest())
        print("ONNX [1,27] + [1,10,27] -> [1,6]; history excludes current")
        print(f"command={self.command}, kp={self.kp}, kd={self.kd}, torque_limit={self.limits}")

    def key(self, key):
        if key in (ord('W'),ord('S')):
            self.command[0] += .1 if key == ord('W') else -.1
        elif key in (ord('A'),ord('D')):
            self.command[1] += .01 if key == ord('A') else -.01
        elif key in (ord('Q'),ord('E')):
            self.command[2] += .1 if key == ord('Q') else -.1
        elif key == 32:
            self.command[[0,2]] = 0
        self.command = np.clip(self.command, self.cfg['command_lower'], self.cfg['command_upper'])

    def state(self):
        q = self.data.qpos[self.qa].copy()
        dq = self.data.qvel[self.va].copy()
        # Match the C++ sensor frame. Gyro is already body-local; do not rotate twice.
        omega = self.data.sensor('trunk_imu_gyro').data.copy()
        quat = self.data.sensor('trunk_imu_quat').data.copy()
        gravity = rotation(quat).T @ [0.,0.,-1.]
        return q, dq, omega, gravity

    def inference(self):
        c = self.cfg
        q, dq, omega, gravity = self.state()
        rel = q - np.array(c['observation_default_dof_pos'])
        rel[WHEELS] = 0
        self.obs = np.concatenate((omega*c['ang_vel_scale'], gravity,
                                   self.command*np.array(c['commands_scale']),
                                   rel*c['dof_pos_scale'], dq*c['dof_vel_scale'], self.action)).astype(np.float32)
        self.obs = np.clip(self.obs, -c['clip_obs'], c['clip_obs'])
        if self.history is None:
            self.history = np.tile(self.obs, (10,1))
        self.last_hist = self.history.copy()
        raw = self.policy.run(['nn_output'], {'nn_input0':self.obs[None],
                                             'nn_input1':self.history[None]})[0][0]
        if not np.isfinite(raw).all():
            raise RuntimeError('Nonfinite ONNX action')
        self.action = np.clip(raw, c['clip_actions_lower'], c['clip_actions_upper']).astype(np.float32)
        # History insertion is AFTER inference, as in current rl_sar configuration.
        self.history[:-1] = self.history[1:]
        self.history[-1] = self.obs
        scaled = self.action*np.array(c['action_scale'])
        self.target = np.clip(self.home+scaled, c['output_pos_lower'], c['output_pos_upper'])
        self.target[WHEELS] = 0
        self.velocity[:] = 0
        self.velocity[WHEELS] = scaled[WHEELS]
        self.velocity = np.clip(self.velocity, c['output_vel_lower'], c['output_vel_upper'])
        tilt = math.degrees(math.acos(float(np.clip(-gravity[2], -1, 1))))
        z = float(self.data.xpos[self.base,2])
        self.tilts.append(tilt); self.heights.append(z); self.actions.append(self.action.copy())
        if self.first_fall is None and (tilt > 60 or z < .12):
            self.first_fall = float(self.data.time)

    def control(self):
        q, dq, _, _ = self.state()
        t = self.data.time
        if t < self.args.prepare:
            # C++ GetUp interpolates from measured pose to home over 1s,
            # then holds that pose for 2s (pre_running_pos == default).
            s = min(t, 1.)
            self.target = self.initial_q + s*(self.home-self.initial_q)
            self.velocity[:] = 0
        self.raw_tau = self.kp*(self.target-q)+self.kd*(self.velocity-dq)
        self.data.ctrl[self.act] = np.clip(self.raw_tau, -self.limits, self.limits)
        if t >= self.args.prepare:
            self.torques.append(self.data.ctrl[self.act].copy())


def run(args):
    prepare_policy(args)
    if args.num_envs != 1:
        print(f"[INFO] MuJoCo runs one robot; --num_envs {args.num_envs} is a compatibility argument.")
    sim = Simulation(args)
    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(sim.model, sim.data, key_callback=sim.key)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.base
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -20
    stream = args.csv.open('w', newline='') if args.csv else None
    writer = csv.writer(stream) if stream else None
    if writer:
        writer.writerow(['sim_time_s']+[f'nn_input0_{i}' for i in range(27)] +
                        [f'nn_input1_{j}_{i}' for j in range(10) for i in range(27)] +
                        [f'nn_output_{i}' for i in range(6)] +
                        [f'qtarget_{i}' for i in range(6)]+[f'dqtarget_{i}' for i in range(6)]+
                        ['base_z','tilt_deg'])
    start = time.monotonic()
    next_pd = 0.; next_rl = args.prepare; next_render = 0.; next_print = 0.
    try:
        while sim.data.time < args.prepare+args.duration:
            # Schedule on simulation time; PD deadlines quantized to physics steps.
            if sim.data.time+1e-9 >= next_rl:
                sim.inference(); next_rl += .02
                if writer:
                    writer.writerow([sim.data.time]+sim.obs.tolist()+sim.last_hist.ravel().tolist()+
                                    sim.action.tolist()+sim.target.tolist()+sim.velocity.tolist()+
                                    [sim.heights[-1],sim.tilts[-1]])
            if sim.data.time+1e-9 >= next_pd:
                sim.control(); next_pd += .005
            mujoco.mj_step(sim.model, sim.data)
            if not np.isfinite(sim.data.qpos).all():
                raise RuntimeError('Nonfinite simulation state')
            if sim.data.time >= next_print:
                print(f't={sim.data.time:.2f} z={sim.data.xpos[sim.base,2]:.3f} '
                      f'cmd={sim.command.round(3)} action={sim.action.round(3)}', flush=True)
                next_print += 1
            if viewer:
                if not viewer.is_running(): break
                if sim.data.time >= next_render:
                    viewer.sync(); next_render += 1/60
                delay = start+sim.data.time-time.monotonic()
                if delay > 0: time.sleep(delay)
    finally:
        if viewer: viewer.close()
        if stream: stream.close()
    if sim.heights:
        result = {'rl_frames':len(sim.heights), 'first_fall_sim_s':sim.first_fall,
                  'policy_sha256':hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  'command':sim.command.tolist(), 'kp':sim.kp.tolist(), 'kd':sim.kd.tolist(),
                  'torque_limit':sim.limits.tolist(),
                  'base_z_min_max':[min(sim.heights),max(sim.heights)],
                  'last_2s_mean_z':float(np.mean(sim.heights[-100:])),
                  'last_2s_tilt_max_deg':max(sim.tilts[-100:]),
                  'tilt_max_deg':max(sim.tilts),
                  'action_abs_max':np.max(np.abs(sim.actions),axis=0).tolist(),
                  'torque_abs_max':np.max(np.abs(sim.torques),axis=0).tolist()}
        print(json.dumps(result, indent=2))
        if args.summary: args.summary.write_text(json.dumps(result,indent=2)+'\n')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=ROOT/'src/rl_sar_zoo/andymini_description/mjcf/scene.xml')
    choice = p.add_mutually_exclusive_group()
    choice.add_argument('--policy', type=Path, help='Use an existing ONNX directly without copying')
    choice.add_argument('--checkpoint', type=Path, help='Training directory or model_N.pt; export and deploy automatically')
    p.add_argument('--task', default='DDT-Height-Flat-Andy-Pace-Robust-Play-v0', help='Isaac task used if export is needed')
    p.add_argument('--num_envs', type=int, default=1, help='Play CLI compatibility; this MuJoCo viewer has one robot')
    p.add_argument('--keyboard', action='store_true', help='Viewer keyboard controls (also enabled by default)')
    p.add_argument('--config',type=Path,default=ROOT/'policy/andymini/robot_lab/config.yaml')
    p.add_argument('--height',type=float,default=.22)
    p.add_argument('--vx',type=float,default=0.)
    p.add_argument('--wz',type=float,default=0.)
    p.add_argument('--duration',type=float,default=120.,help='RL duration after preparation')
    p.add_argument('--prepare',type=float,default=3.)
    p.add_argument('--dt',type=float,default=.002)
    for name in ('kp','kd','wheel-kd','torque-limit'):
        p.add_argument('--'+name,type=float)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--csv',type=Path)
    p.add_argument('--summary',type=Path)
    a = p.parse_args()
    if a.num_envs < 1:
        p.error("--num_envs must be positive")
    if not all(math.isfinite(v) for v in (a.height,a.vx,a.wz,a.dt,a.duration,a.prepare)) or not 0<a.dt<=.005 or a.duration<=0 or a.prepare<0:
        p.error('Invalid timing or command')
    for name in ('kp','kd','wheel_kd','torque_limit'):
        v=getattr(a,name)
        if v is not None and (not math.isfinite(v) or v<0):p.error('Invalid gain/limit')
    return a


if __name__ == '__main__':
    run(parse_args())
