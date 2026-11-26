#!/usr/bin/env python3
"""Simulated flame (digital) and MQ-2 smoke (analog) sensors.

Publishes the same topics and value formats the real Arduino bridge will use:
  /flame_detected  std_msgs/Bool   True = flame seen (already active-high)
  /smoke_level     std_msgs/Int32  raw 10-bit ADC reading, 0-1023 (Arduino A0)

Readings are computed from the robot pose on /odom and the hazard positions
given as parameters. Positions are in the odom frame, which equals the world
frame when the robot spawns at the world origin facing +x.
"""
import math
import random

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Int32


def pairs(flat):
    """[x1, y1, x2, y2, ...] -> [(x1, y1), (x2, y2), ...]"""
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class VirtualSensorsNode(Node):
    def __init__(self):
        super().__init__('virtual_sensors_node')

        # Hazard positions as flat [x1, y1, x2, y2, ...] lists (metres)
        self.declare_parameter('fire_positions', [3.0, 0.0])
        self.declare_parameter('smoke_positions', [3.0, 0.0])

        # Flame sensor: short range, directional cone
        self.declare_parameter('flame_range', 1.5)      # m
        self.declare_parameter('flame_fov_deg', 60.0)   # total cone angle

        # MQ-2 smoke sensor: non-directional, slow, noisy
        self.declare_parameter('smoke_baseline', 120.0)  # clean-air ADC reading
        self.declare_parameter('smoke_peak', 700.0)      # reading at the source
        self.declare_parameter('smoke_radius', 2.5)      # m, spread of the cloud
        self.declare_parameter('smoke_tau', 1.5)         # s, sensor response lag
        self.declare_parameter('smoke_noise', 5.0)       # ADC counts (std dev)

        self.declare_parameter('rate_hz', 10.0)

        gp = self.get_parameter
        self.fires = pairs(gp('fire_positions').value)
        self.smokes = pairs(gp('smoke_positions').value)
        self.flame_range = float(gp('flame_range').value)
        self.flame_half_fov = math.radians(float(gp('flame_fov_deg').value)) / 2.0
        self.baseline = float(gp('smoke_baseline').value)
        self.peak = float(gp('smoke_peak').value)
        self.radius = float(gp('smoke_radius').value)
        self.tau = float(gp('smoke_tau').value)
        self.noise = float(gp('smoke_noise').value)
        rate = float(gp('rate_hz').value)
        self.dt = 1.0 / rate

        self.pose = None            # (x, y, yaw)
        self.smoke_filtered = self.baseline
        self.last_flame = False

        self.flame_pub = self.create_publisher(Bool, '/flame_detected', 10)
        self.smoke_pub = self.create_publisher(Int32, '/smoke_level', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_timer(self.dt, self.update)

        self.get_logger().info(
            f'Virtual sensors started: {len(self.fires)} fire(s), '
            f'{len(self.smokes)} smoke source(s).'
        )

    def odom_callback(self, msg):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.pose = (p.position.x, p.position.y, yaw)

    def flame_seen(self):
        x, y, yaw = self.pose
        for fx, fy in self.fires:
            dist = math.hypot(fx - x, fy - y)
            bearing = wrap_angle(math.atan2(fy - y, fx - x) - yaw)
            if dist <= self.flame_range and abs(bearing) <= self.flame_half_fov:
                return True
        return False

    def smoke_target(self):
        x, y, _ = self.pose
        value = self.baseline
        for sx, sy in self.smokes:
            dist = math.hypot(sx - x, sy - y)
            value += (self.peak - self.baseline) * math.exp(-((dist / self.radius) ** 2))
        return value

    def update(self):
        if self.pose is None:
            # No odometry yet: behave like a sensor in clean air.
            flame = False
            target = self.baseline
        else:
            flame = self.flame_seen()
            target = self.smoke_target()

        # First-order lag: the MQ-2 reacts slowly, it does not jump.
        alpha = self.dt / (self.tau + self.dt)
        self.smoke_filtered += alpha * (target - self.smoke_filtered)
        reading = self.smoke_filtered + random.gauss(0.0, self.noise)
        reading = int(round(min(1023.0, max(0.0, reading))))

        self.flame_pub.publish(Bool(data=flame))
        self.smoke_pub.publish(Int32(data=reading))

        if flame != self.last_flame:
            self.get_logger().info(f'Flame sensor: {"FLAME" if flame else "clear"}')
            self.last_flame = flame


def main(args=None):
    rclpy.init(args=args)
    node = VirtualSensorsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
