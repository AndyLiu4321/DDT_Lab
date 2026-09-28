"""PACE fixed-height flat-ground deployment training: physics/PD/policy=1000/200/50 Hz."""
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from .robust_env_cfg import AndyPaceRobustHeightFlatEnvCfg
from .fixed_height_mdp import FixedHeightCommand, stationary_pose
from .sampled_actuator import SampledPaceDCMotorCfg


@configclass
class AndyPaceFixedHeightFlatEnvCfg(AndyPaceRobustHeightFlatEnvCfg):
    """Inherits AndyFlatEnvCfg through PACE; independent from existing Height tasks."""
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.class_type = FixedHeightCommand
        self.commands.base_velocity.base_height = (.35, .35)
        self.commands.base_velocity.extreme_low_height = .35
        self.commands.base_velocity.standing_fraction = .5
        self.commands.base_velocity.resampling_time_range = (5., 10.)
        # Remove the height-dependent joint-pose guide and low-height contact mode.
        self.rewards.tracking_leg_2_height = None
        self.rewards.auxiliary_contact_low = None
        self.rewards.stationary_pose = RewardTermCfg(func=stationary_pose, weight=2.)
        self.rewards.stationary_translation.weight = -4.
        self.rewards.stationary_yaw.weight = -1.
        self.rewards.stationary_tracking.weight = 2.
        self.rewards.flat_orientation_l2.weight = -5.
        self.sim.dt = .001
        self.decimation = 20
        for name in ('legs', 'drive_wheels'):
            old = self.scene.robot.actuators[name]
            new = SampledPaceDCMotorCfg()
            for key, value in vars(old).items():
                if key != 'class_type':
                    setattr(new, key, value)
            new.control_decimation = 5
            new.max_delay = 5  # Additional sampled-torque transport delay: 0--5 ms.
            new.strength_range = (.8, 1.)
            self.scene.robot.actuators[name] = new
        # Broad, bounded ranges centered on the PACE candidate. These are training
        # hypotheses, not confidence intervals measured on the real robot.
        gains = self.events.randomize_actuator_gains.params
        gains['stiffness_distribution_params'] = (.8, 1.2)
        gains['damping_distribution_params'] = (.8, 1.2)
        joint = self.events.pace_joint_parameters.params
        joint['friction_distribution_params'] = (.6, 1.4)
        joint['armature_distribution_params'] = (.75, 1.25)
        self.events.add_base_mass.params['mass_distribution_params'] = (.85, 1.15)
        self.events.add_base_com.params['com_range'] = {axis: (-.01, .01) for axis in ('x', 'y', 'z')}
        # Restore rigid-body inertia randomization using the base task event.
        import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
        from isaaclab.managers import EventTermCfg, SceneEntityCfg
        self.events.add_base_inertia = EventTermCfg(
            func=mdp.randomize_rigid_body_inertia, mode='startup', params={
                'asset_cfg': SceneEntityCfg('robot', body_names='.*'),
                'inertia_distribution_params': (.85, 1.15), 'operation': 'scale'})
        self.events.physics_material.params.update(
            static_friction_range=(.4, 1.2), dynamic_friction_range=(.3, 1.),
            restitution_range=(0., .1))
        self.events.reset_robot_joints.params.update(
            position_range=(-.08, .08), velocity_range=(-.2, .2))
        pose = self.events.reset_base.params['pose_range']
        pose.update(roll=(-.08, .08), pitch=(-.08, .08))
        noise = self.observations.policy
        noise.base_ang_vel.noise = UniformNoiseCfg(n_min=-.2, n_max=.2)
        noise.projected_gravity.noise = UniformNoiseCfg(n_min=-.04, n_max=.04)
        noise.joint_pos.noise = UniformNoiseCfg(n_min=-.01, n_max=.01)
        noise.joint_vel.noise = UniformNoiseCfg(n_min=-.5, n_max=.5)
        self.events.push_robot.interval_range_s = (5., 10.)
        # Existing performance-gated curriculum starts at zero and reaches .2 m/s.


@configclass
class AndyPaceFixedHeightFlatEnvCfg_PLAY(AndyPaceFixedHeightFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        # Default playback is a nominal check, not a randomized stress test.
        for name in ('randomize_actuator_gains', 'pace_joint_parameters', 'add_base_mass',
                     'add_base_com', 'add_base_inertia', 'physics_material', 'push_robot'):
            setattr(self.events, name, None)
        self.curriculum.pace_push = None
        self.observations.policy.enable_corruption = False
        for name in ('legs', 'drive_wheels'):
            self.scene.robot.actuators[name].max_delay = 0
            self.scene.robot.actuators[name].strength_range = (1., 1.)
