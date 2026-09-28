from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class AndyMiniRoughCfg(LeggedRobotCfg):
    """基于 Aug26_09-03-36_，使用实机允许的 PD 增益。"""

    seed = 1

    # 该开关复现 Aug26 run 的“实际执行逻辑”：当时配置快照虽然开启了
    # COM、电机强度、Kp、Kd 随机化，但 AndyMini 控制器尚未真正使用它们。
    # 设为 False 可恢复当前完整域随机化以及偏航指令死区。
    reproduce_aug26_behavior = True

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        # Six policy actions control four leg joints and two drive wheels. The
        # two auxiliary wheels remain passive physical DOFs.
        num_actions = 6
        # ang_vel(3), gravity(3), commands(3), and four values per action.
        num_observations = 33
        num_obs_hist = 5
        # obs(33), base velocity(3), 9 body contact forces(27), heights(187)
        num_privileged_obs = 250

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = False
        # 独立于连续采样区间的极低姿态高度。
        extreme_low_height = 0.22

        class ranges:
            # command[0]：机体前进速度，单位 m/s。
            lin_vel_x = [-0.6, 0.6]
            # command[1]：目标基座高度，单位 m。
            base_height = [0.22, 0.37]
            # command[2]：目标偏航角速度，单位 rad/s。
            ang_vel_yaw = [-0.6, 0.6]
            heading = [-3.14, 3.14]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        # plane trimesh
        # plane 地形在该 run 的最终运行参数中关闭课程学习。
        curriculum = False
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = True
        measured_points_x = [
            -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        ]
        measured_points_y = [
            -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
            0.1, 0.2, 0.3, 0.4, 0.5,
        ]
        selected = False
        terrain_kwargs = None
        
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.35, 0.15, 0.20, 0.20, 0.10]
        max_init_terrain_level = 5
        difficulty_scale = 0.5
        slope_treshold = 0.75

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.34]
        # default_joint_angles = {
        #     "joint_left_leg_1": -0.3,
        #     "joint_left_leg_2": 1.18,
        #     "joint_left_leg_3": 0.0,
        #     "joint_left_wheel_1": 0.0,
        #     "joint_right_leg_1": -0.3,
        #     "joint_right_leg_2": 1.18,
        #     "joint_right_leg_3": 0.0,
        #     "joint_right_wheel_1": 0.0,
        # }
        default_joint_angles = {
            "joint_left_leg_1": 0,
            "joint_left_leg_2": 0.66,
            "joint_left_leg_3": 0.0,
            "joint_left_wheel_1": 0.0,
            "joint_right_leg_1": 0,
            "joint_right_leg_2": 0.66,
            "joint_right_leg_3": 0.0,
            "joint_right_wheel_1": 0.0,
        }
        # 0.22 m 极低姿态的关节参考。轮子是连续旋转关节，
        # 其角度不作为训练跟踪目标，但保留在完整姿态定义中。
        extreme_low_joint_angles = {
            "joint_left_leg_1": 0.0,
            "joint_left_leg_2": 0.0,
            "joint_left_leg_3": 0.0,
            "joint_left_wheel_1": 0.0,
            "joint_right_leg_1": 0.0,
            "joint_right_leg_2": 0.0,
            "joint_right_leg_3": 0.0,
            "joint_right_wheel_1": 0.0,
        }
        # default_joint_angles = {
        #     "joint_left_leg_1": 0.0,
        #     "joint_left_leg_2": 0.0,
        #     "joint_left_leg_3": 0.0,
        #     "joint_left_wheel_1": 0.0,
        #     "joint_right_leg_1": 0.0,
        #     "joint_right_leg_2": 0.0,
        #     "joint_right_leg_3": 0.0,
        #     "joint_right_wheel_1": 0.0,
        # }
        init_joint_angles = default_joint_angles.copy()

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = { #Kp
            "leg_1": 40.0,
            "leg_2": 40.0,
            # leg_3 is the driven wheel and uses velocity control. 
            "leg_3": 0.0,
        }
        damping = { #Kd
            "leg_1": 1, # 1->0.1
            "leg_2": 1, # 1->0.1
            "leg_3": 0.5,  # 0.5->0.05
        }
        action_scale = 0.15
        # leg_2 以 0.66 rad 为动作中心；动作 -1 对应 0 rad。
        # 正向目标超过 URDF 上限时会被下方的硬限位裁剪。
        leg_2_action_scale = 0.66
        # 位置目标必须远离 URDF 硬限位，避免 PD 持续顶住机械限位。 0.05
        position_limit_margin = 0.0
        vel_scale = 10.0
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            "andymini/urdf/andymini.urdf"
        )
        name = "andymini"
        # leg_3 links are driven wheels; wheel_1 links are passive auxiliaries.
        foot_name = "leg_3"
        wheel_name = ["leg_3"]
        penalize_contacts_on = ["leg_1", "leg_2", "base_link"]
        # 除主动轮 leg_3 和辅助轮 wheel_1 外，其余机体碰地均重置。
        terminate_after_contacts_on = ["base_link", "leg_1", "leg_2"]
        self_collisions = 0
        collapse_fixed_joints = True
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        # 以下数值来自 Aug26_09-03-36_/training_parameters.json。
        rand_interval_s = 4
        randomize_friction = True
        friction_range = [0.6, 1.2]
        randomize_base_mass = True
        added_mass_range = [-0.25, 0.25]
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_Kp_factor = True
        Kp_factor_range = [0.9, 1.1]
        randomize_Kd_factor = True
        Kd_factor_range = [0.9, 1.1]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_action_latency = False
        latency_range = [0.0, 0.0]

    class normalization(LeggedRobotCfg.normalization):
        # 显式固定该 run 的裁剪和观测缩放，避免基类修改影响复现。
        contact_force_range = [0.0, 50.0]
        clip_observations = 100.0
        clip_actions = 100.0

        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class rewards(LeggedRobotCfg.rewards):
        # 将终止奖励以外的总奖励裁剪为非负值，避免训练早期惩罚占主导。
        only_positive_rewards = True
        # 速度跟踪奖励的指数核宽度；数值越小，对跟踪误差越敏感。
        tracking_sigma = 0.25
        # 0.22 m 极低姿态需要 leg_2 到达 URDF 下限 0 rad，
        # 因此不再将位置目标缩到 95% 软限位内。
        soft_dof_pos_limit = 1.0
        # 软关节速度限制相对于 URDF velocity limit 的比例。
        soft_dof_vel_limit = 1.0
        # 软力矩限制相对于 URDF effort limit 的比例。
        soft_torque_limit = 1.0
        # 动态高度奖励使用 exp(-误差平方/sigma)，0.0025 对应 0.05 m。
        height_tracking_sigma = 0.0025
        # 仅 0.22 m 极低姿态使用辅助轮，更高时要求辅助轮离地。
        auxiliary_contact_height_threshold = 0.22
        # 接触力模长超过该值即判定辅助轮接地，单位为牛顿。
        auxiliary_contact_force_threshold = 1.0
        # 0.23--0.37 m 区间内，leg_2 目标随高度线性伸腿。
        leg_2_low_target = 0.05
        leg_2_high_target = 1.18
        # leg_2 目标误差约 0.2 rad 时，引导奖励下降到 exp(-1)。
        leg_2_tracking_sigma = 0.04
        # 基座过低仍作为独立的安全终止条件。
        min_base_height = 0.12
        # 允许的最大足端接触力，超过该值时可用于计算接触力惩罚，单位为牛顿。
        max_contact_force = 80.0

        class scales(LeggedRobotCfg.rewards.scales):
            # 非超时终止惩罚，例如基座触地导致的提前结束。
            termination = -1.0
            # 奖励机器人跟踪目标平面线速度。
            tracking_lin_vel = 3.0
            # 奖励机器人跟踪目标偏航角速度。
            tracking_ang_vel = 1.0
            # 惩罚基座沿竖直方向的速度，减少上下跳动。
            lin_vel_z = -1.0
            # 惩罚基座横滚和俯仰角速度，提高运动稳定性。
            ang_vel_xy = -0.1
            # 惩罚基座偏离水平姿态。
            # 加强水平姿态约束，避免为了接触辅助轮而长期俯仰或侧倾。
            orientation = -5.0
            # 惩罚关节力矩平方和，降低能耗和过大的控制输出。
            torques = -0.00001
            # 惩罚非轮式关节速度平方和；主动轮速度在机器人类中被排除。
            dof_vel = -1e-7
            # 惩罚受控关节加速度，抑制高频抖动。
            dof_acc = -5e-7
            # 禁用继承的固定目标高度惩罚，使用 command[1] 动态高度。
            base_height = 0.0
            tracking_base_height = 6.0
            # 日志绝对值除以 10 约为平均高度误差（m）。
            base_height_error = -10.0
            # 低高度模式奖励辅助轮接地；每侧轮分别贡献一半。
            auxiliary_contact_low = 2.0
            # 高高度模式惩罚辅助轮接地，避免始终依赖辅助轮行驶。
            auxiliary_contact_high = -2.0
            # 引导 leg_2 根据高度指令收腿或伸腿，帮助从零探索高度控制。
            tracking_leg_2_height = 2.0
            # 惩罚左右腿偏离镜像对称姿态。
            joint_symmetry = -2.0
            # 奖励轮足具有适当的离地时间，鼓励跨越地形。
            # feet_air_time = 0.2
            # 惩罚配置中指定的腿部或基座发生非期望碰撞。
            collision = -1.0
            # 惩罚轮足侧向撞击障碍物或地形边缘。
            feet_stumble = -0.2
            # 惩罚相邻控制周期的动作变化，提升动作平滑性。
            action_rate = -0.04
            # 在零速度指令下惩罚腿关节偏离默认姿态。
            stand_still = 0.0
            # 惩罚关节位置超过软限位。
            dof_pos_limits = -0.2
            # 下面三项在 Aug26 run 中由基类继承且实际处于启用状态。
            feet_air_time = 0.1
            joint_power = -2e-5
            power_distribution = -1e-5


class AndyMiniRoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1

    class policy(LeggedRobotCfgPPO.policy):
        # leg_2 已有较大的独立动作范围，降低初始噪声以减少限位撞击。
        init_noise_std = 0.5
        activation = "elu"
        # Keep stochastic VAE samples bounded during very long training runs.
        vae_sigma_min = 0.01
        vae_sigma_max = 1.0

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # 防止错误局部最优中策略噪声持续增长，同时保留必要探索。
        entropy_coef = 0.005
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        vae_learning_rate = 3e-4
        kl_weight = 0.1

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic_DWAQ"
        algorithm_class_name = "PPO"
        run_name = ""
        experiment_name = "rough_andymini"
        num_steps_per_env = 24
        max_iterations = 30000
        load_run = -1
        checkpoint = -1
        resume = False
        resume_path = -1
