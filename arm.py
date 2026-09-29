"""
Manipulation helpers:
  - pixel_to_chassis: from a camera pixel to the object's 3D position relative to the car
  - inverse_kinematics: "put the gripper at this point" → 4 joint angles
  - PickSequence: the sequence of motions that picks up the apple

Coordinates are in the chassis frame: x = forward, y = left, z = up, in meters.
"""
import numpy as np

# Arm dimensions (from picar.xml)
YAW_AXIS_X = 0.03       # x position of arm_base on the chassis (the yaw axis passes here)
SHOULDER_Z = 0.086      # height of the shoulder joint: arm_base (0.051) + upper_arm (0.035)
UPPER_ARM = 0.12        # shoulder to elbow
FOREARM = 0.10          # elbow to wrist
WRIST_TO_GRIP = 0.07    # wrist to the point between the fingers (grip_site)

GRIPPER_OPEN = 0.025
GRIPPER_CLOSED = 0.0

# Where the apple is carried: in front of the car, centered, ~14 cm above the ground.
# (Lifting straight up from where the apple lies is unreachable — the stretched-out arm can't go higher.)
CARRY_POINT = (0.17, 0.0, 0.08)


def pixel_to_chassis(model, data, camera, u, v, width, height, ground_z):
    """Intersect the ray through pixel (u, v) with the horizontal plane z = ground_z.
    Returns that point's (x, y, z) relative to the car."""
    fovy = np.deg2rad(model.cam(camera).fovy[0])
    focal = (height / 2) / np.tan(fovy / 2)          # focal length in pixels
    # A MuJoCo camera looks along its own -z; image right = +x, image up = +y
    ray_camera = np.array([(u - width / 2) / focal, -(v - height / 2) / focal, -1.0])
    camera_rotation = data.cam(camera).xmat.reshape(3, 3)
    ray_world = camera_rotation @ ray_camera
    camera_pos = data.cam(camera).xpos
    t = (ground_z - camera_pos[2]) / ray_world[2]     # how far along the ray the plane is
    point_world = camera_pos + t * ray_world

    chassis_rotation = data.body("chassis").xmat.reshape(3, 3)
    return chassis_rotation.T @ (point_world - data.body("chassis").xpos)


def inverse_kinematics(x, y, z):
    """Joint angles (arm_yaw, shoulder, elbow, wrist) that put the point between the fingers
    at (x, y, z) with the gripper pointing straight down. Raises ValueError if unreachable."""
    # 1) arm_yaw: turn the arm toward the target. After that the problem is planar (2D)
    yaw = np.arctan2(y, x - YAW_AXIS_X)
    reach = np.hypot(x - YAW_AXIS_X, y)               # horizontal distance from the yaw axis

    # 2) The gripper points down, so the wrist is WRIST_TO_GRIP above the target
    dr = reach                                         # shoulder to wrist: horizontal
    dz = z + WRIST_TO_GRIP - SHOULDER_Z                # shoulder to wrist: vertical
    distance_sq = dr ** 2 + dz ** 2

    # 3) Two-link arm: elbow angle from the law of cosines
    cos_elbow = (distance_sq - UPPER_ARM ** 2 - FOREARM ** 2) / (2 * UPPER_ARM * FOREARM)
    if abs(cos_elbow) > 1:
        raise ValueError(f"Point is out of reach: ({x:.3f}, {y:.3f}, {z:.3f})")
    elbow = np.arccos(cos_elbow)

    # 4) Shoulder: direction to the wrist (measured from vertical) minus the angle added by the elbow bend
    shoulder = np.arctan2(dr, dz) - np.arctan2(FOREARM * np.sin(elbow), UPPER_ARM + FOREARM * np.cos(elbow))

    # 5) Wrist: when the three pitch angles add up to 90°, the gripper points straight down
    wrist = np.pi / 2 - shoulder - elbow
    return np.array([yaw, shoulder, elbow, wrist])


class PickSequence:
    """Pick-up motions: each step = (duration s, arm joint angles, gripper opening).
    The arm moves smoothly (linearly) from one step's pose to the next."""

    def __init__(self, target, start_pose, start_gripper):
        x, y, z = target
        above = inverse_kinematics(x, y, z + 0.08)    # 8 cm above the target
        grasp = inverse_kinematics(x, y, z + 0.007)   # fingers squeeze the apple around its middle
        lift = inverse_kinematics(x, y, z + 0.08)     # first lift straight up 8 cm with the apple
        carry = inverse_kinematics(*CARRY_POINT)      # then bring it to the front, centered
        self.steps = [
            (2.0, above, GRIPPER_OPEN),     # open and move above the target
            (1.5, grasp, GRIPPER_OPEN),     # descend
            (1.0, grasp, GRIPPER_CLOSED),   # squeeze
            (1.5, lift, GRIPPER_CLOSED),    # lift straight up (don't drag it along the floor)
            (1.5, carry, GRIPPER_CLOSED),   # move to the carry pose
        ]
        self.step_index = 0
        self.step_time = 0.0
        self.from_pose = np.array(start_pose, dtype=float)
        self.from_gripper = start_gripper

    @property
    def finished(self):
        return self.step_index >= len(self.steps)

    def update(self, dt):
        """Advance by dt seconds and return the (arm joint angles, gripper) command."""
        if self.finished:
            _, pose, gripper = self.steps[-1]
            return pose, gripper
        duration, pose, gripper = self.steps[self.step_index]
        self.step_time += dt
        a = min(1.0, self.step_time / duration)          # 0 → 1: how much of this step is done
        command_pose = self.from_pose + (pose - self.from_pose) * a
        command_gripper = self.from_gripper + (gripper - self.from_gripper) * a
        if a >= 1.0:
            self.step_index += 1
            self.step_time = 0.0
            self.from_pose, self.from_gripper = pose, gripper
        return command_pose, command_gripper
