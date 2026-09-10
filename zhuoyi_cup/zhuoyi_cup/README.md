# 卓翼杯 2026 全国集群智能技术挑战赛

本文档面向参赛队伍，说明比赛环境启动方式、公开 ROS 接口、参考 Demo 运行方法，以及比赛中的关键约束。

`/home/ubuntu/zhuoyi_cup` 是组委会提供的比赛运行包。平台负责启动 RflySim/UE、CopterSim、PX4 SITL、rostrans 和裁判系统；参赛程序通过公开 ROS 话题读取目标/本机信息，并向拦截机和吊舱发送控制指令。

## 比赛流程

参赛队伍在比赛镜像中开发和测试算法。正式比赛时，参赛队伍按要求提交可运行的参赛代码；组委会按公开 ROS 接口接入参赛程序，并运行指定赛道和难度。

比赛分为 `3v3` 和 `10v10` 两个赛道。每个赛道分别运行 `low`、`mid`、`high` 三个难度，三局成绩合计作为该赛道成绩。

## 目录说明

比赛包默认部署在 `/home/ubuntu/zhuoyi_cup`，下表路径均相对于该目录。

| 相对路径 | 说明 |
| --- | --- |
| `run.sh` | 启动一局比赛。 |
| `stop.sh` | 停止比赛和相关后台进程。 |
| `README.md` | 说明文档。 |
| `demo/contestant_demo_3v3.py` | 参赛接口参考 Demo。 |
| `config/` | 比赛启动配置，包括场景、吊舱/视觉、rostrans 和 UE 碰撞配置。 |
| `logs/` | 运行日志目录。发现平台问题时，附上问题说明并把对应日志目录发给组委会。 |
| `sim/` | 仿真平台启动脚本，`run.sh` 会自动调用 `start_coptersim.sh` 等脚本。 |
| `bin/` | 裁判运行程序，包含 `competition_runtime`。参赛队不需要直接运行。 |

## 快速运行

进入比赛包目录后，选择赛道和难度启动一局：

```bash
cd /home/ubuntu/zhuoyi_cup
./run.sh --scene 3v3 --diff low
```

平台启动后，可在 UE 窗口中按住 `B` 并按数字键切换无人机视角。每架无人机头顶显示编号，编号颜色用于区分双方：拦截机编号为红色，靶机编号为蓝色。`3v3` 中 1-3 为拦截机、4-6 为靶机；`10v10` 中 1-10 为拦截机、11-20 为靶机。

比赛计时从任一拦截机解锁开始。500 秒后打击不再计入成绩。当全部靶机被击中、所有剩余靶机到达安全区，或达到 500 秒上限时，裁判冻结成绩，并持续显示最终打击数和最后打击时间。

参数说明：

| 参数 | 可选值 | 说明 |
| --- | --- | --- |
| `--scene` | `3v3`、`10v10` | 比赛赛道。3v3 开启可见光吊舱；10v10 不提供图像和吊舱接口。 |
| `--diff` | `low`、`mid`、`high` | 靶机突防难度。`low` 匀速定高，`mid` 匀速变高，`high` 变速变高。 |
| `--seed` | 数字 | 可选。固定随机种子，便于复现实验。正式比赛以组委会指定方式运行。 |

看到 `run.sh` 终端打印 `初始化成功` 后，保持该终端运行，另开一个终端运行 3v3 参考 Demo：

```bash
cd /home/ubuntu/zhuoyi_cup
python3 demo/contestant_demo_3v3.py --uav 1
```

该 Demo 演示公开话题订阅、拦截机位置/速度/偏航控制，以及吊舱角速度和回中控制。

停止比赛可以在运行终端按 `Ctrl-C`，也可以另开终端执行：

```bash
./stop.sh
```

每次启动会生成 `logs/<时间>_<scene>_<diff>/`，并将 `logs/latest` 指向最新一局。日志主要用于组委会定位平台问题；如果发现平台异常，请附上问题说明，并把该次日志目录发给组委会。

## 公开接口

坐标约定：所有公开位置均使用 ENU 语义，`x=East`、`y=North`、`z=Up`。所有带 `/mavros` 的话题接口规范均与 MAVROS 官方定义一致。

3v3 赛道发布 3 架拦截机和 3 架靶机相关话题；10v10 赛道扩展到 10 架拦截机和 10 架靶机，且不发布吊舱图像、吊舱状态和吊舱控制话题。

参赛队伍只允许通过公开 ROS 话题与比赛系统交互。任何试图绕开 ROS 话题，使用其他方法与比赛系统或裁判系统通信交互的行为，均视为违规。

### 无人机控制

| 话题/接口 | 类型 | 方向 | 说明 |
| --- | --- | --- | --- |
| `/interceptor{i}/origin` | `geometry_msgs/PointStamped` | 平台 → 参赛代码 | 第 `i` 架拦截机世界 ENU 出生点。 |
| `/radar/target{i}/position` | `geometry_msgs/PointStamped` | 平台 → 参赛代码 | 第 `i` 架靶机世界 ENU 位置。3v3 为带噪位置（1 Hz），噪声幅度为 low 10m、mid 15m、high 15m；10v10 为靶机雷达位置（10 Hz）。靶机被击毁后停止发布该话题。 |
| `/mavros{i}/local_position/odom` | `nav_msgs/Odometry` | 平台 → 参赛代码 | 第 `i` 架拦截机局部 ENU 里程计，坐标原点为该无人机初始位置。 |
| `/mavros{i}/state` | `mavros_msgs/State` | 平台 → 参赛代码 | 拦截机连接、解锁、飞控模式状态。 |
| `/mavros{i}/setpoint_raw/local` | `mavros_msgs/PositionTarget` | 参赛代码 → 平台 | 拦截机 setpoint 控制。 |
| `/rflysim{i}/imu` | `sensor_msgs/Imu` | 平台 → 参赛代码 | 第 `i` 架拦截机 IMU 数据。 |

1 号拦截机命名空间为 `/mavros`，2 号起为 `/mavros2`、`/mavros3`，10v10 以此类推到 `/mavros10`。拦截机控制使用 MAVROS `PositionTarget`，发送到对应命名空间的 `setpoint_raw/local`：

- 1 号拦截机：`/mavros/setpoint_raw/local`
- 2 号拦截机：`/mavros2/setpoint_raw/local`
- 3 号拦截机：`/mavros3/setpoint_raw/local`
- 10v10 以此类推到 `/mavros10/setpoint_raw/local`

局部 ENU 原点为该机出生点，轴与世界 ENU 平行。因此如果要飞向某个世界点：

```text
local_target = world_target - /interceptor{i}/origin
```

`/mavros{i}/setpoint_raw/local` 使用官方 MAVROS `mavros_msgs/PositionTarget` 协议。参赛代码可以使用位置、速度、偏航等字段，具体 `type_mask` 用法参考 `demo/contestant_demo_3v3.py`。

### 吊舱状态

吊舱接口仅在 3v3 赛道提供。`sensorN` 与第 `N` 架拦截机对应，例如 `sensor1` 对应 1 号拦截机。

| 话题 | 类型 | 方向 | 说明 |
| --- | --- | --- | --- |
| `/rflysim/sensor{N}/img_cine` | `sensor_msgs/Image` | 平台 → 参赛代码 | 第 `N` 个吊舱的可见光图像。 |
| `/rflysim/sensor{N}/gimbal_status` | `rflysim_msgs/GimbalStatus` | 平台 → 参赛代码 | 第 `N` 个吊舱的相机内参、安装外参和当前姿态。 |

状态话题为 `/rflysim/sensor{N}/gimbal_status`，类型为 `rflysim_msgs/GimbalStatus`。当前字段如下：

```text
std_msgs/Header header
float32 fx
float32 fy
float32 cx
float32 cy
float32 k1
float32 k2
float32 p1
float32 p2
float32 k3
float32 p2g_roll_deg
float32 p2g_pitch_deg
float32 p2g_yaw_deg
float32 p2g_front_m
float32 p2g_right_m
float32 p2g_down_m
float32 g2c_roll_deg
float32 g2c_pitch_deg
float32 g2c_yaw_deg
uint8 source_id
```

比赛中主要关注：

| 字段 | 说明 |
| --- | --- |
| `fx, fy, cx, cy` | 相机内参，单位为像素。 |
| `k1, k2, p1, p2, k3` | 畸变参数。 |
| `p2g_*` | 机体系到吊舱安装外参，包含角度和前/右/下方向平移。 |
| `g2c_roll_deg, g2c_pitch_deg, g2c_yaw_deg` | 吊舱到相机的当前姿态角。 |
| `source_id` | 吊舱来源编号。 |

图像测量和目标定位通常重点使用 `fx/fy/cx/cy`、畸变参数、`p2g_*`、`g2c_*` 和 `source_id`。

### 吊舱控制

吊舱控制接口仅在 3v3 赛道提供。

控制话题为 `/rflysim/sensor{N}/gimbal_ctrl`，类型为 `rflysim_msgs/GimbalCtrl`。当前消息定义如下：

```text
uint8 CTRL_ATTITUDE_SPEED=4
std_msgs/Header header
uint8 ctrl_type
bool back_to_center
float32 yaw_speed_dps
float32 pitch_speed_dps
```

当前支持以下控制方式：

| 控制 | 设置方式 |
| --- | --- |
| yaw 角速度 | `ctrl_type=4`，设置 `yaw_speed_dps`，单位 deg/s。 |
| pitch 角速度 | `ctrl_type=4`，设置 `pitch_speed_dps`，单位 deg/s。 |
| 回中 | `ctrl_type=4`，设置 `back_to_center=true`。 |
| 停止转动 | `ctrl_type=4`，将 `yaw_speed_dps=0`、`pitch_speed_dps=0`。 |

参赛队伍可直接参考 `demo/contestant_demo_3v3.py` 中的吊舱控制示例。

## 比赛规则要点

3v3 赛道：

- 3 架拦截机对 3 架靶机。
- 基于模糊雷达位置抵近，进入末端后使用可见光吊舱感知与跟踪。
- 初/中/高三种难度各运行 1 次，满分拦截数为 9。

10v10 赛道：

- 10 架拦截机对 10 架靶机。
- 不提供吊舱图像，使用雷达位置进行协同决策和拦截。
- 初/中/高三种难度各运行 1 次，满分拦截数为 30。

### 比赛启动与靶机突围触发

靶机开机后在地面待命、不解锁。检测到**任一拦截机解锁**（比赛计时同时开始）后，全体靶机起飞并各自爬升到预设悬停高度；**最后一架靶机到达并稳定悬停**后，全体靶机同时进入突围模式，朝安全区突防并按难度做相应机动；一旦触发不再返回待机。

排名先按总拦截成功数排序；拦截数相同，按总用时短者靠前。单局最大计分时长为 500 秒。

正式比赛期间，参赛队伍不得修改 `/home/ubuntu/zhuoyi_cup` 目录内的任何文件。

## 问题反馈

如果发现平台 bug 或异常现象，请记录现象、运行命令和对应日志目录，并及时反馈给组委会。不得利用平台 bug、接口漏洞或非公开通信方式获取比赛收益；此类行为视为违规。

