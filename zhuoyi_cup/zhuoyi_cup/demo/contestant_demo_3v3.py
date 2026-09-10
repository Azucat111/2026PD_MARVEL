#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""卓翼杯 3v3 参赛队接口参考 Demo。

这是给参赛队伍看的最小可运行示例, 展示三件事:
  1) 公开话题怎么订阅、数据长什么样:
       /interceptorX/origin           拦截机世界 ENU 出生点(恒定)
       /mavrosX/local_position/odom   拦截机局部 ENU 里程计(原点=出生点)
       /mavrosX/state                 解锁/飞控模式
       /radar/targetN/position        雷达目标位置(3v3 带噪)
       /rflysim/sensorX/gimbal_status 吊舱状态(内参/吊舱角/source_id)
       /rflysim/sensorX/img_cine      吊舱图像(本 demo 只统计收到, 不解码)
  2) 无人机控制(发到 /mavrosX/setpoint_raw/local): 位置控制 / 速度控制 / 偏航;
  3) 吊舱控制(发到 /rflysim/sensorX/gimbal_ctrl): 角速度转向(ctrl=4)+回中。
默认只对 1 号拦截机 + 1 号吊舱做控制演示, 其余两架照样订阅监看。

只在云端跑(需 ROS + mavros + rflysim_msgs)。先 source SDK 再运行:
    source /opt/rostrans/sdk/x86_64-u20.04-ros1-noetic/setup.bash
    python3 demo/contestant_demo_3v3.py [--uav 1]

坐标约定(关键, 务必看懂):
  - X 号拦截机的 MAVROS 命名空间: 1 号是 /mavros(无后缀), 2/3 号是 /mavros2 /mavros3。
  - /interceptorX/origin = 该机世界 ENU 出生点(恒定); /mavrosX/.../odom = 局部 ENU(原点=出生点, 轴与世界平行无旋转)。
  - 控制设定点发到 /mavrosX/setpoint_raw/local, 用**局部 ENU**(coordinate_frame=LOCAL_NED, mavros 内部转 PX4):
        位置设定点 local = 世界目标点 - 出生点;   速度设定点 = 世界速度(两系轴平行, 直接用)。
  - 吊舱 sensorX 与 X 号拦截机对应(sensor1=拦截机1)。
"""
import argparse
import math
import sys
import time

# ROS/SDK 依赖只在云端有: 放进 try, 且**不在导入期** sys.exit, 否则本地 import 本文件就崩。
try:
    import rospy
    from geometry_msgs.msg import PointStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    from mavros_msgs.msg import PositionTarget, State
    from mavros_msgs.srv import CommandBool, SetMode
    from rflysim_msgs.msg import GimbalCtrl, GimbalStatus
    _ROS_OK = True
except Exception as _exc:        # noqa: BLE001
    _ROS_OK = False
    _ROS_ERR = _exc

# PositionTarget.type_mask 位: 0~2=x/y/z位置, 3~5=vx/vy/vz速度, 6~8=加速度, 10=yaw, 11=yaw_rate。
# 位置控制: 只用位置 + yaw(忽略 速度/加速度/yaw_rate)。
POSITION_ONLY_TYPEMASK = (1 << 3 | 1 << 4 | 1 << 5 | 1 << 6 | 1 << 7 | 1 << 8 | 1 << 11)
# 速度控制: 只用速度 + yaw(忽略 位置/加速度/yaw_rate)。
VELOCITY_ONLY_TYPEMASK = (1 << 0 | 1 << 1 | 1 << 2 | 1 << 6 | 1 << 7 | 1 << 8 | 1 << 11)
FRAME_LOCAL_NED = 1


def mavros_ns(uav):
    """X 号拦截机的 MAVROS 命名空间: 1 号是 /mavros, 其余是 /mavrosX。"""
    return "/mavros" if int(uav) == 1 else f"/mavros{int(uav)}"


class PublicTopics:
    """订阅**所有公开话题**, 缓存每个话题的最新数据(参赛队就是这样拿数据的)。"""

    def __init__(self, uav_ids=(1, 2, 3), target_ids=(1, 2, 3), sensor_ids=(1, 2, 3)):
        self.origin = {}          # interceptorX/origin: 世界 ENU 出生点
        self.odom = {}            # mavrosX/local_position/odom: 局部 ENU (x,y,z)
        self.state = {}           # mavrosX/state: (armed, mode)
        self.radar = {}           # radar/targetN/position: 世界 ENU
        self.gimbal = {}          # sensorX/gimbal_status: GimbalStatus
        self.img_count = {}       # sensorX/img_cine: 收到帧数
        for i in uav_ids:
            ns = mavros_ns(i)
            rospy.Subscriber(f"/interceptor{i}/origin", PointStamped,
                             lambda m, k=i: self.origin.__setitem__(k, (m.point.x, m.point.y, m.point.z)))
            rospy.Subscriber(f"{ns}/local_position/odom", Odometry,
                             lambda m, k=i: self.odom.__setitem__(
                                 k, (m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z)))
            rospy.Subscriber(f"{ns}/state", State,
                             lambda m, k=i: self.state.__setitem__(k, (bool(m.armed), m.mode)))
        for j in target_ids:
            rospy.Subscriber(f"/radar/target{j}/position", PointStamped,
                             lambda m, k=j: self.radar.__setitem__(k, (m.point.x, m.point.y, m.point.z)))
        for s in sensor_ids:
            rospy.Subscriber(f"/rflysim/sensor{s}/gimbal_status", GimbalStatus,
                             lambda m, k=s: self.gimbal.__setitem__(k, m))
            rospy.Subscriber(f"/rflysim/sensor{s}/img_cine", Image,
                             lambda m, k=s: self.img_count.__setitem__(k, self.img_count.get(k, 0) + 1))

    def dump(self, title="公开话题当前数据"):
        """打印一次公开话题概览。"""
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        for i in sorted(set(self.origin) | set(self.odom) | set(self.state)):
            o = self.origin.get(i); od = self.odom.get(i); st = self.state.get(i)
            print(f"拦截机{i}: 出生点={_fmt(o)}  局部里程计={_fmt(od)}  "
                  f"state={'已解锁' if st and st[0] else '未解锁'}/{st[1] if st else '?'}")
        for j in sorted(self.radar):
            print(f"雷达目标{j}: 世界ENU={_fmt(self.radar[j])}")
        for s in sorted(set(self.gimbal) | set(self.img_count)):
            g = self.gimbal.get(s)
            gi = (f"内参=({g.fx:.0f},{g.fy:.0f},{g.cx:.0f},{g.cy:.0f}) "
                  f"g2c(yaw={g.g2c_yaw_deg:+.1f}, pitch={g.g2c_pitch_deg:+.1f}) "
                  f"source_id={g.source_id}" if g else "无")
            print(f"吊舱{s}: status[{gi}]  图像帧数={self.img_count.get(s, 0)}")

    def one_line(self, uav):
        """给控制阶段打印一行状态。"""
        st = self.state.get(uav)
        od = self.odom.get(uav)
        return f"state={'已解锁' if st and st[0] else '未解锁'}/{st[1] if st else '?'} local={_fmt(od)}"


def _fmt(p):
    return "无" if p is None else f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})"


# ---------------------------- 无人机控制 ----------------------------

def make_pos_setpoint(local_enu, yaw=0.0):
    """位置设定点(局部 ENU): 飞到该点并对准 yaw。local = 世界目标 - 出生点。"""
    msg = PositionTarget()
    msg.coordinate_frame = FRAME_LOCAL_NED
    msg.type_mask = POSITION_ONLY_TYPEMASK
    msg.position.x, msg.position.y, msg.position.z = local_enu
    msg.yaw = float(yaw)
    return msg


def make_vel_setpoint(vel_enu, yaw=0.0):
    """速度设定点(局部 ENU=世界 ENU, 轴平行): 按该速度飞, 机头对准 yaw。"""
    msg = PositionTarget()
    msg.coordinate_frame = FRAME_LOCAL_NED
    msg.type_mask = VELOCITY_ONLY_TYPEMASK
    msg.velocity.x, msg.velocity.y, msg.velocity.z = vel_enu
    msg.yaw = float(yaw)
    return msg


class DroneDemo:
    """对单架拦截机演示: 解锁 -> OFFBOARD -> 起飞 -> 位置/速度/偏航控制。"""

    def __init__(self, uav, topics, verbose=False):
        self.uav = uav
        self.ns = mavros_ns(uav)
        self.topics = topics
        self.verbose = bool(verbose)
        self.pub = rospy.Publisher(f"{self.ns}/setpoint_raw/local", PositionTarget, queue_size=10)
        # 解锁/切模式走 mavros 服务; 若云端(rostrans)没暴露这俩服务, 优雅降级为"只发设定点",
        # 假定飞机已被外部解锁(也方便参赛队按自己的方式解锁)。
        self._arm = self._mode = None
        try:
            rospy.wait_for_service(f"{self.ns}/cmd/arming", timeout=5.0)
            rospy.wait_for_service(f"{self.ns}/set_mode", timeout=5.0)
            self._arm = rospy.ServiceProxy(f"{self.ns}/cmd/arming", CommandBool)
            self._mode = rospy.ServiceProxy(f"{self.ns}/set_mode", SetMode)
        except rospy.ROSException:
            print(f"[UAV{uav}] 提示: 找不到 {self.ns}/cmd/arming 或 set_mode 服务, "
                  f"将只持续发设定点(请自行确保已解锁+OFFBOARD)。")

    def cur_local(self):
        return self.topics.odom.get(self.uav, (0.0, 0.0, 0.0))

    def stream(self, make_msg, duration_s, rate_hz=20.0, label=""):
        """以 rate_hz 持续发设定点 duration_s 秒(PX4 OFFBOARD 必须 >2Hz 持续发)。

        make_msg() 每拍重新生成一条设定点(可用 self.cur_local() 做闭环)。
        """
        if label:
            print(f"[UAV{self.uav}] 开始: {label}")
        rate = rospy.Rate(rate_hz)
        t_end = time.time() + duration_s
        last_print = 0.0
        while time.time() < t_end and not rospy.is_shutdown():
            self.pub.publish(make_msg())
            if self.verbose and time.time() - last_print > 1.0:
                print(f"[UAV{self.uav}]   {self.topics.one_line(self.uav)}")
                last_print = time.time()
            rate.sleep()
        if label:
            print(f"[UAV{self.uav}] 完成: {label}; {self.topics.one_line(self.uav)}")

    def arm_and_offboard(self):
        # 1) 先持续发零速设定点, 飞控才允许进 OFFBOARD; 2) 再解锁 + 切模式(各重试几次)。
        self.stream(lambda: make_vel_setpoint((0, 0, 0)), 1.5, label="预热设定点流(进 OFFBOARD 前必须先发)")
        if self._arm is None:        # 没有 mavros 服务: 只发设定点, 假定外部已解锁。
            self.stream(lambda: make_vel_setpoint((0, 0, 0)), 1.0, label="无解锁服务, 仅维持设定点流")
            return
        for _ in range(5):
            try:
                self._arm(True)
                self._mode(custom_mode="OFFBOARD")
            except rospy.ServiceException as exc:
                print(f"[UAV{self.uav}] arm/set_mode 调用异常，重试: {exc}")
            st = self.topics.state.get(self.uav)
            if st and st[0] and st[1] == "OFFBOARD":
                print(f"[UAV{self.uav}] 已解锁并进入 OFFBOARD")
                return
            self.stream(lambda: make_vel_setpoint((0, 0, 0)), 0.5)
        print(f"[UAV{self.uav}] 警告: 未确认 OFFBOARD/解锁, 仍继续演示; {self.topics.one_line(self.uav)}")

    def takeoff(self, alt=30.0):
        # 起飞用【速度控制】爬升: 综合模型/rostrans 对速度设定点响应可靠;
        # 直接发位置设定点从地面起飞常常不爬(位置控制更适合空中已起飞后用, 见下方 demo_position_control)。
        def climb():
            cz = self.cur_local()[2]
            vu = max(-2.0, min(3.0, 1.0 * (alt - cz)))   # 朝 alt 的垂直速度(限幅 ±2~3 m/s)
            return make_vel_setpoint((0.0, 0.0, vu))
        self.stream(climb, 12.0, label=f"起飞到局部高度 {alt:.0f}m(速度控制爬升)")

    def demo_position_control(self):
        x0, y0, z0 = self.cur_local()
        # 【位置控制】飞到当前点东侧 15m(ENU: x=东). 想飞向某世界目标: local = 世界目标 - /interceptor 出生点。
        self.stream(lambda: make_pos_setpoint((x0 + 15.0, y0, z0)), 6.0,
                    label="无人机控制①【位置控制】向东平移 15m(发 position 设定点)")

    def demo_velocity_control(self):
        # 【速度控制】以 (0,3,0) m/s 向北飞 3s, 再 (0,0,0) 悬停 2s。
        self.stream(lambda: make_vel_setpoint((0.0, 3.0, 0.0)), 3.0,
                    label="无人机控制②【速度控制】向北 3 m/s 飞 3s(发 velocity 设定点)")
        self.stream(lambda: make_vel_setpoint((0.0, 0.0, 0.0)), 2.0, label="    速度归零, 悬停")

    def demo_yaw_control(self):
        # 【偏航】速度设定点带 yaw: 机头转到东(yaw=+90°, 从北顺时针). 边低速前进边转头。
        self.stream(lambda: make_vel_setpoint((1.0, 0.0, 0.0), yaw=math.radians(90.0)), 4.0,
                    label="无人机控制③【偏航】机头转向正东(yaw=90°), setpoint 的 yaw 字段)")


# ---------------------------- 吊舱控制 ----------------------------

class GimbalDemo:
    """对单个吊舱演示: 角速度转向(ctrl=4)+回中。"""

    def __init__(self, sensor, topics):
        self.sensor = sensor
        self.topics = topics
        self.pub = rospy.Publisher(f"/rflysim/sensor{sensor}/gimbal_ctrl", GimbalCtrl, queue_size=10)

    def _g2c(self):
        g = self.topics.gimbal.get(self.sensor)
        return (g.g2c_yaw_deg, g.g2c_pitch_deg) if g else (0.0, 0.0)

    def _stream(self, msg, duration_s, rate_hz=30.0, label=""):
        if label:
            print(f"[Sensor{self.sensor}] 开始: {label}")
        rate = rospy.Rate(rate_hz)
        t_end = time.time() + duration_s
        while time.time() < t_end and not rospy.is_shutdown():
            msg.header.stamp = rospy.Time.now()
            self.pub.publish(msg)
            rate.sleep()
        y, p = self._g2c()
        print(f"[Sensor{self.sensor}] 完成: {label}; g2c yaw={y:+.1f}° pitch={p:+.1f}°")

    def _rate_msg(self, yaw_dps=0.0, pitch_dps=0.0, recenter=False):
        msg = GimbalCtrl()
        msg.ctrl_type = GimbalCtrl.CTRL_ATTITUDE_SPEED        # =4
        msg.yaw_speed_dps = float(yaw_dps)
        msg.pitch_speed_dps = float(pitch_dps)
        msg.back_to_center = bool(recenter)
        return msg

    def demo_attitude_speed(self):
        # 吊舱控制①【角速度转向 ctrl=4】: yaw 速率左转 -> 回中 -> pitch 速率抬头 -> 回中。
        self._stream(self._rate_msg(yaw_dps=-25.0), 2.0, label="吊舱 yaw 左转 25°/s")
        self._stream(self._rate_msg(recenter=True), 1.5, label="吊舱回中")
        self._stream(self._rate_msg(pitch_dps=15.0), 2.0, label="吊舱 pitch 抬头 15°/s")
        self._stream(self._rate_msg(recenter=True), 1.5, label="吊舱回中")


def run(uav, verbose=False):
    rospy.init_node("contestant_demo_3v3", anonymous=True)
    topics = PublicTopics()
    print("=" * 72)
    print(f"卓翼杯 3v3 参赛接口 Demo: UAV{uav} / Sensor{uav}")
    print("=" * 72)
    print("等待公开话题首帧: origin / odom / state / radar / gimbal ...")
    t_end = time.time() + 15.0
    while time.time() < t_end and not rospy.is_shutdown():
        if uav in topics.odom and uav in topics.origin and topics.gimbal:
            break
        time.sleep(0.2)
    topics.dump("阶段0: 公开话题数据概览")

    print("\n" + "=" * 72)
    print("阶段1: 拦截机控制演示")
    print("=" * 72)
    drone = DroneDemo(uav, topics, verbose=verbose)
    drone.arm_and_offboard()
    drone.takeoff(alt=30.0)
    drone.demo_position_control()
    drone.demo_velocity_control()
    drone.demo_yaw_control()
    drone.stream(lambda: make_vel_setpoint((0, 0, 0)), 2.0, label="演示间悬停")

    print("\n" + "=" * 72)
    print("阶段2: 吊舱控制演示")
    print("=" * 72)
    gimbal = GimbalDemo(uav, topics)      # sensorX 与 X 号拦截机对应
    if uav in topics.gimbal:
        gimbal.demo_attitude_speed()
    else:
        print(f"[Sensor{uav}] 收不到 gimbal_status, 跳过吊舱演示(检查 UE/rostrans 与 sensor 编号)")

    print("\n" + "=" * 72)
    print("演示结束: 末尾悬停，Ctrl-C 退出")
    print("=" * 72)
    drone.stream(lambda: make_vel_setpoint((0, 0, 0)), 5.0, label="末尾悬停")


def main():
    p = argparse.ArgumentParser(description="卓翼杯 3v3 参赛队接口参考 Demo")
    p.add_argument("--uav", type=int, default=1, choices=[1, 2, 3],
                   help="对哪架拦截机+吊舱做控制演示(其余照样订阅监看); 默认 1")
    p.add_argument("--verbose", action="store_true", help="控制过程中每秒打印一次本机状态")
    argv = rospy.myargv()[1:] if _ROS_OK else sys.argv[1:]
    if not _ROS_OK:
        if any(arg in ("-h", "--help") for arg in argv):
            p.parse_args(argv)
            return 0
        print(f"Error: 无法导入 ROS/mavros/rflysim_msgs: {_ROS_ERR}\n"
              f"先 source /opt/rostrans/sdk/x86_64-u20.04-ros1-noetic/setup.bash 再运行。")
        return 1
    args = p.parse_args(argv)
    try:
        run(args.uav, verbose=args.verbose)
    except rospy.ROSInterruptException:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
