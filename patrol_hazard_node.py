#!/usr/bin/env python3
"""Hazard monitor with autonomous patrol and safety-assisted teleop.

Always running, in both modes:
  - watches smoke (/smoke_level), flame (/flame_detected) and the camera
  - publishes alerts on /hazard_alert (confidence = how many sources agree)

mode:=auto (default)
  - patrol forward a bounded distance, slow down near hazards
  - stop and HOLD at a hazard; call /return_home to reverse back to the start
mode:=manual
  - you drive with teleop remapped to /cmd_vel_teleop; this node forwards it
    to /cmd_vel but blocks FORWARD motion while a hazard is active

This node is the only publisher on /cmd_vel.
"""
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger

CONFIDENCE = {1: 'LOW', 2: 'MEDIUM', 3: 'HIGH'}


class PatrolHazardNode(Node):
    def __init__(self):
        super().__init__('patrol_hazard_node')
        self.bridge = CvBridge()

        self.declare_parameter('mode', 'auto')               # 'auto' or 'manual'
        self.declare_parameter('patrol_speed', 0.2)          # m/s
        self.declare_parameter('slow_speed', 0.1)            # m/s near hazards
        self.declare_parameter('patrol_distance', 5.0)       # m from the start point
        self.declare_parameter('standoff_distance', 1.0)     # m, camera-based stop distance
        self.declare_parameter('camera_slow_distance', 2.0)  # m, slow down inside this
        self.declare_parameter('marker_width', 0.2)          # m, real marker width
        self.declare_parameter('focal_px', 554.0)            # 640 px, 1.047 rad FOV
        self.declare_parameter('min_area', 500.0)            # px^2, ignore tiny blobs
        self.declare_parameter('smoke_warn', 300)            # raw ADC warning level
        self.declare_parameter('smoke_clear_margin', 30)     # hysteresis below warn level
        gp = self.get_parameter
        self.mode = str(gp('mode').value).lower()
        if self.mode not in ('auto', 'manual'):
            self.get_logger().warn(f'Unknown mode "{self.mode}", using "auto".')
            self.mode = 'auto'
        self.speed = float(gp('patrol_speed').value)
        self.slow_speed = float(gp('slow_speed').value)
        self.patrol_distance = float(gp('patrol_distance').value)
        self.standoff = float(gp('standoff_distance').value)
        self.cam_slow_dist = float(gp('camera_slow_distance').value)
        self.marker_width = float(gp('marker_width').value)
        self.focal_px = float(gp('focal_px').value)
        self.min_area = float(gp('min_area').value)
        self.smoke_warn = int(gp('smoke_warn').value)
        self.smoke_margin = int(gp('smoke_clear_margin').value)

        # PATROLLING, HAZARD_HOLD, RETURNING, COMPLETE (auto) or MANUAL
        self.state = 'PATROLLING' if self.mode == 'auto' else 'MANUAL'
        self.pose = None            # (x, y, yaw) from /odom
        self.start = None           # (x, y) where the node first saw odometry

        self.flame = False
        self.smoke = 0
        self.sensors_seen = False
        self.smoke_elevated = False

        self.cam = None             # latest camera detection (dict) or None
        self.last_frame = None
        self.teleop = None          # (linear_x, angular_z, time) from /cmd_vel_teleop

        self.hazard_latched = False  # True during one hazard "episode"
        self.clear_since = None

        # Jazzy's diff_drive_controller expects TwistStamped on /cmd_vel.
        # The stamp is left empty on purpose; the controller fills it in.
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.alert_pub = self.create_publisher(String, '/hazard_alert', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Bool, '/flame_detected', self.flame_callback, 10)
        self.create_subscription(Int32, '/smoke_level', self.smoke_callback, 10)
        self.create_subscription(
            TwistStamped, '/cmd_vel_teleop', self.teleop_callback, 10
        )
        self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, qos_profile_sensor_data
        )
        self.create_service(Trigger, 'return_home', self.return_home_callback)
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info(f'Hazard node started in {self.mode.upper()} mode.')

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------- callbacks ----------------
    def odom_callback(self, msg):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.pose = (p.position.x, p.position.y, yaw)
        if self.start is None:
            self.start = (p.position.x, p.position.y)

    def flame_callback(self, msg):
        self.flame = bool(msg.data)
        self.sensors_seen = True

    def smoke_callback(self, msg):
        self.smoke = int(msg.data)
        self.sensors_seen = True

    def teleop_callback(self, msg):
        self.teleop = (msg.twist.linear.x, msg.twist.angular.z, self.now_sec())

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e!r}')
            return
        self.last_frame = frame
        self.cam = None

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Red wraps around the hue axis, so use two ranges.
        mask = cv2.inRange(hsv, (0, 120, 100), (10, 255, 255)) | cv2.inRange(
            hsv, (170, 120, 100), (180, 255, 255)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return

        x, y, bw, bh = cv2.boundingRect(largest)
        distance = (self.focal_px * self.marker_width) / bw
        hx = hy = None
        if self.pose is not None:
            # Image right is the robot's right (negative yaw direction).
            bearing = -math.atan2((x + bw / 2.0) - w / 2.0, self.focal_px)
            px, py, yaw = self.pose
            hx = px + distance * math.cos(yaw + bearing)
            hy = py + distance * math.sin(yaw + bearing)
        self.cam = {
            't': self.now_sec(),
            'distance': distance,
            'x': hx,
            'y': hy,
            'bbox': (x, y, bw, bh),
        }

    def return_home_callback(self, request, response):
        if self.mode == 'auto' and self.state == 'HAZARD_HOLD':
            self.state = 'RETURNING'
            response.success = True
            response.message = 'Returning to start (reversing).'
        else:
            response.success = False
            response.message = (
                f'Only available in auto mode during HAZARD_HOLD '
                f'(mode={self.mode}, state={self.state}).'
            )
        self.get_logger().info(response.message)
        return response

    # ---------------- monitoring (runs in every mode and state) ----------------
    def update_smoke_warning(self):
        """Early warning with hysteresis so it does not flap."""
        if not self.smoke_elevated and self.smoke >= self.smoke_warn:
            self.smoke_elevated = True
            text = f'EARLY_WARNING sensor=smoke level={self.smoke}'
            self.get_logger().warn(text)
            self.alert_pub.publish(String(data=text))
        elif self.smoke_elevated and self.smoke < self.smoke_warn - self.smoke_margin:
            self.smoke_elevated = False

    def update_hazard_episode(self, now, hazard, cam):
        """One alert per episode; the episode ends after 2 s with no hazard."""
        if hazard:
            self.clear_since = None
            if not self.hazard_latched:
                self.hazard_latched = True
                self.raise_alert(now, cam)
        elif self.hazard_latched:
            if self.clear_since is None:
                self.clear_since = now
            elif now - self.clear_since > 2.0:
                self.hazard_latched = False
                self.get_logger().info('Hazard cleared.')

    def raise_alert(self, now, cam):
        sources = []
        if self.flame:
            sources.append('flame')
        if cam:
            sources.append('camera')
        if self.smoke_elevated:
            sources.append('smoke')
        confidence = CONFIDENCE[len(sources)]

        if cam and cam['x'] is not None:
            where = f'x={cam["x"]:.2f} y={cam["y"]:.2f} distance={cam["distance"]:.2f}'
        elif cam:
            where = f'distance={cam["distance"]:.2f}'
        elif self.pose is not None:
            where = f'near_robot x={self.pose[0]:.2f} y={self.pose[1]:.2f}'
        else:
            where = 'position=unknown'
        text = (
            f'HAZARD confidence={confidence} sources={"+".join(sources)} '
            f'smoke={self.smoke} flame={self.flame} {where}'
        )
        self.get_logger().warn(text)
        self.alert_pub.publish(String(data=text))

        if self.last_frame is not None:
            frame = self.last_frame.copy()
            if cam:
                x, y, bw, bh = cam['bbox']
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                frame, f'{confidence}: {"+".join(sources)}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            path = f'/tmp/hazard_{int(now)}.png'
            cv2.imwrite(path, frame)
            self.get_logger().info(f'Snapshot saved: {path}')

    # ---------------- control ----------------
    def control_loop(self):
        now = self.now_sec()
        cam = self.cam if (self.cam and now - self.cam['t'] < 1.0) else None
        if not self.sensors_seen:
            self.get_logger().warn(
                'No data on /flame_detected or /smoke_level: camera-only mode.',
                throttle_duration_sec=10.0,
            )

        self.update_smoke_warning()
        hazard = self.flame or (cam is not None and cam['distance'] <= self.standoff)
        self.update_hazard_episode(now, hazard, cam)
        slow = (
            self.smoke_elevated
            or self.flame
            or (cam is not None and cam['distance'] <= self.cam_slow_dist)
        )

        if self.mode == 'manual':
            lin, ang = self.manual_command(now, hazard, slow)
        else:
            lin, ang = self.auto_speed(hazard, slow), 0.0

        # Always publish, so the robot is actively commanded to stop too.
        cmd = TwistStamped()
        cmd.twist.linear.x = float(lin)
        cmd.twist.angular.z = float(ang)
        self.cmd_pub.publish(cmd)

    def manual_command(self, now, hazard, slow):
        lin = ang = 0.0
        t = self.teleop
        if t is not None and now - t[2] < 0.5:
            lin, ang = t[0], t[1]
        if lin > 0.0:
            if hazard:
                lin = 0.0
                self.get_logger().warn(
                    'Forward blocked: hazard ahead. Reverse or turn away.',
                    throttle_duration_sec=2.0,
                )
            elif slow:
                lin = min(lin, self.slow_speed)
        return lin, ang

    def auto_speed(self, hazard, slow):
        if self.pose is None:
            self.get_logger().warn(
                'Waiting for /odom before moving...', throttle_duration_sec=2.0
            )
            return 0.0
        dist_from_start = math.hypot(
            self.pose[0] - self.start[0], self.pose[1] - self.start[1]
        )

        if self.state == 'PATROLLING':
            if hazard:
                self.state = 'HAZARD_HOLD'
                self.get_logger().warn(
                    'Hazard: holding position. Call /return_home to retreat.'
                )
                return 0.0
            if dist_from_start >= self.patrol_distance:
                self.state = 'COMPLETE'
                self.get_logger().info(
                    f'Patrol route finished: {dist_from_start:.1f} m from start '
                    f'(limit {self.patrol_distance:.1f} m).'
                )
                return 0.0
            return self.slow_speed if slow else self.speed

        if self.state == 'RETURNING':
            if dist_from_start <= 0.15:
                self.state = 'COMPLETE'
                self.get_logger().info('Returned to start.')
                return 0.0
            return -self.slow_speed

        return 0.0  # HAZARD_HOLD or COMPLETE


def main(args=None):
    rclpy.init(args=args)
    node = PatrolHazardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()