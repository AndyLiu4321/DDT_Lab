"""Andy flat-ground height control: [forward velocity, height, yaw rate]."""

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import ddt_lab.tasks.manager_based.locomotion.mdp as mdp

from . import height_mdp
from .flat_env_cfg import AndyFlatEnvCfg
from .rough_env_cfg import ANDY_AUXILIARY_WHEEL_BODIES, ANDY_POSITION_JOINTS


@configclass
class HeightCommandsCfg:
    # 指令为 [前进速度 vx (m/s), 基座高度 (m), 偏航角速度 (rad/s)]。
    # 名称沿用 base_velocity，以兼容现有 NP3O 观测和回放流程。
    # 范围、重采样周期和高度端点采样比例见 height_mdp.py 的 HeightCommandCfg/HeightCommand。
    base_velocity = height_mdp.HeightCommandCfg()


@configclass
class AndyHeightFlatEnvCfg(AndyFlatEnvCfg):
    commands: HeightCommandsCfg = HeightCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 继承 Andy 平地场景、六维动作、NP3O 观测历史和成本约束。
        # 位置软限位使用完整关节范围，让最低姿态可以到达 leg_2 的 0 rad。
        self.scene.robot.soft_joint_pos_limit_factor = 1.0
        # 四个腿部关节的位置 PD 基准增益；训练时还会受到增益随机化影响。
        self.scene.robot.actuators["legs"].stiffness = 40.0
        self.scene.robot.actuators["legs"].damping = 1.0
        # 动作顺序：左 leg_1/leg_2/驱动轮，右 leg_1/leg_2/驱动轮。
        # 位置目标 = 默认关节角 + 动作 × scale，再裁剪到关节限位。
        # leg_1 尺度 0.15 rad；leg_2 尺度 0.66 rad，动作 -1 对应目标 0 rad。
        for name, scale in (("joint_pos_0", 0.15), ("joint_pos_1", 0.66),
                            ("joint_pos_3", 0.15), ("joint_pos_4", 0.66)):
            action = getattr(self.actions, name)
            action.class_type = height_mdp.LimitedJointPositionAction
            action.scale = scale
        # 两个驱动轮使用速度控制：单位动作对应 10 rad/s；辅助轮保持被动。
        self.actions.joint_vel_2.scale = 10.0
        self.actions.joint_vel_5.scale = 10.0
        # Actor/Critic 的三维指令分别乘以 (2, 5, 0.25)，第二维为高度而非横向速度。
        for group in (self.observations.policy, self.observations.critic):
            group.velocity_commands.scale = (2.0, 5.0, 0.25)

        # 仅跟踪前进速度，避免把高度指令误当成 vy；偏航仍使用原跟踪奖励。
        self.rewards.track_lin_vel_xy_exp = RewTerm(func=height_mdp.tracking_forward, weight=3.0)
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        # 稳定性惩罚：竖直速度、横滚/俯仰角速度、关节加速度和动作变化。
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.dof_acc_l2.weight = -5e-7
        self.rewards.action_rate_l2.weight = -0.1
        # 约束左右腿对称，并惩罚非超时终止。
        self.rewards.joint_mirror.weight = -2.0
        self.rewards.is_terminated.weight = -1.0
        # 关闭固定高度与恢复姿态奖励，辅助轮接触改为按目标高度分模式处理。
        # 高度不参与零速度判定；按实际运动惩罚，不强迫回到默认关节角。
        self.rewards.stand_still = None
        self.rewards.stationary_motion = RewTerm(
            func=height_mdp.stationary_motion, weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANDY_POSITION_JOINTS)},
        )
        self.rewards.base_height_l2 = None
        self.rewards.upward = None
        self.rewards.upright_progress = None
        self.rewards.auxiliary_wheel_contacts = None
        # 动态高度奖励：6 × exp(-高度误差平方 / 0.0025)，另加 -10 × 绝对误差。
        # 高度相对平地环境原点计算；误差 0.05 m 时，指数项为 exp(-1)。
        self.rewards.tracking_base_height = RewTerm(func=height_mdp.tracking_height, weight=6.0)
        self.rewards.base_height_error = RewTerm(func=height_mdp.height_error, weight=-10.0)
        # 目标高度 <= 0.22 m 时奖励辅助轮接地；更高时惩罚辅助轮接地。
        # 接触力模长 > 1 N 判定接触；两侧辅助轮各贡献一半。
        for name, low, weight in (("auxiliary_contact_low", True, 2.0),
                                  ("auxiliary_contact_high", False, -2.0)):
            setattr(self.rewards, name, RewTerm(
                func=height_mdp.auxiliary_contact, weight=weight,
                params={"low": low, "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=ANDY_AUXILIARY_WHEEL_BODIES)},
            ))
        # 姿态引导：正常高度下 leg_2 目标随高度从 0.05 线性增至 1.18 rad。
        # 最低模式 (<= 0.22 m) 则引导四个位置关节全部回到 0 rad。
        self.rewards.tracking_leg_2_height = RewTerm(
            func=height_mdp.tracking_leg_height, weight=0.5,
            params={"leg_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_2"]),
                    "position_cfg": SceneEntityCfg("robot", joint_names=ANDY_POSITION_JOINTS)},
        )
        # 仅惩罚四个位置关节的速度与位置越限，排除连续旋转的驱动轮。
        self.rewards.dof_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2, weight=-0.001,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANDY_POSITION_JOINTS)},
        )
        self.rewards.dof_pos_limits = RewTerm(
            func=mdp.joint_pos_limits, weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANDY_POSITION_JOINTS)},
        )
        # 基座或非驱动腿段接地时终止；驱动轮和辅助轮允许接地。
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["base_link", ".*_leg_(1|2)"])
        # 平地基座高度低于 0.12 m 时终止，阈值默认值见 height_mdp.below_min_height。
        self.terminations.min_base_height = DoneTerm(func=height_mdp.below_min_height)

        # 域随机化继承自 mini/rough_env_cfg.py 的 EventCfg，Andy 对关节选择器做了适配：
        # startup（创建环境时采样，此后保持）：
        #   physics_material：所有刚体静/动摩擦 0.2–1.6，恢复系数 0–1，64 桶。
        #   add_base_mass：仅 base_link 默认质量加 [-0.5, 2.0] kg。
        #   add_base_inertia：所有刚体惯量对角分量乘 [0.9, 1.1]。
        #   add_base_com：所有刚体质心 XYZ 分别偏移 [-0.05, 0.05] m。
        # reset（每次重置时采样）：
        #   randomize_actuator_gains：六个受控关节 Kp/Kd 乘 [0.8, 1.2]，对数均匀分布。
        #   reset_base：XY 偏移 ±0.5 m，偏航 ±3.14 rad；各轴线/角速度 ±0.5。
        #   reset_robot_joints：默认关节角乘 [-0.5, 1.0] 后裁剪到限位，关节速度为 0。
        # base_external_force_torque 和 push_robot 已在 AndyRoughEnvCfg 中关闭。
        # 策略观测噪声在父类 ObservationsCfg 中配置，训练默认开启。
        #
        # 只修改此高度任务时，在本方法内覆盖对应事件参数，例如：
        # self.events.add_base_mass.params["mass_distribution_params"] = (-0.25, 0.25)
        # self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        # 关闭某项：self.events.add_base_com = None
        # 添加事件：导入 EventTermCfg，设置 self.events.<名称> = EventTermCfg(...)。
        # mode="startup"/"reset"/"interval" 分别表示创建时、重置时、周期触发；
        # 周期事件还需要 interval_range_s=(最小间隔秒数, 最大间隔秒数)。




@configclass
class AndyHeightFlatEnvCfg_PLAY(AndyHeightFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 回放默认 50 个环境；play.py 的命令行/键盘模式可进一步覆盖数量。
        self.scene.num_envs = 50
        # 回放关闭观测噪声及质量、惯量、质心、PD 增益随机化。
        # 接触材质随机化和初始状态随机化仍继承训练配置。
        self.observations.policy.enable_corruption = False
        self.events.add_base_mass = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.randomize_actuator_gains = None
