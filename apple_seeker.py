"""
The robot's "brain": finds the red apple with the front camera, drives to it and picks it up with the arm.

States (state machine):
    search   — no apple in view: drive a figure-eight — one full circle left, then one full circle right.
               Circling in only one direction never sees an apple sitting inside the turning circle
               (the camera always faces outward along the circle); on the opposite circle it lies outside.
               The figure-eight also keeps the car near where it started.
    approach — apple in view: steer toward it (P-controller) and drive closer
    backup   — apple is close but off-center (out of the arm's reach): back up a bit and re-align
    reached  — apple is close and centered: stop and compute its 3D position from the camera
    picking  — the arm is grasping and lifting the apple
    holding  — the apple is in the gripper

Run:   python apple_seeker.py
Keys (click a window first):  r = move the apple to a new random spot,  q = quit
"""
import cv2
import mujoco
import numpy as np

from arm import GRIPPER_CLOSED, PickSequence, pixel_to_chassis
from vision import draw_apple, find_apple

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
STEPS_PER_FRAME = 16     # 16 * 0.002 s = 0.032 s → the brain decides ~30 times per second
FRAME_TIME = STEPS_PER_FRAME * 0.002

SEARCH_SPEED = 5.0       # wheel speed while searching (rad/s)
SEARCH_STEERING = 0.5    # full steering lock while searching — the car drives in a circle
CIRCLE_FRAMES = 441      # one full circle: measured 441 frames = 14.1 s (radius ~0.49 m, the wheels slip a bit)
APPROACH_SPEED = 10.0    # approach speed when the apple is far
MIN_APPROACH_SPEED = 3.0 # slow down when the apple is very close
STEER_GAIN = 1.0         # P-controller gain: error 1 (apple at the image edge) → steering 1.0 (clipped to 0.5)
STOP_AREA = 9000         # apple bigger than this many pixels = we've arrived (~8 cm from the bumper)
CENTER_TOLERANCE = 0.3   # when stopping, the apple must be this close to the image center (else out of reach)
BACKUP_SPEED = 5.0       # reverse speed while re-aligning
BACKUP_FRAMES = 40       # ~1.3 s of reversing
LOST_PATIENCE = 10       # frames without the apple before we call it lost (tolerates one-frame misses)
APPLE_RADIUS = 0.02

model = mujoco.MjModel.from_xml_path("picar.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
MAX_STEERING = model.actuator("steering").ctrlrange[1]
ARM_JOINTS = ("arm_yaw", "shoulder", "elbow", "wrist")

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

APPLE_QPOS = model.jnt_qposadr[model.body("apple").jntadr[0]]  # where the apple's freejoint lives in qpos
APPLE_QVEL = model.jnt_dofadr[model.body("apple").jntadr[0]]


def clamp(value, low, high):
    return max(low, min(high, value))


def new_memory():
    """The brain's memory."""
    return {"state": "search", "lost_frames": 0, "last_steering": 0.0, "last_error": 0.0,
            "search_frames": 0, "backup_frames": 0, "last_apple": None}


def search(memory):
    """Search pattern: figure-eight — CIRCLE_FRAMES frames turning left, then as many turning right, and so on."""
    memory["search_frames"] += 1
    circle_number = memory["search_frames"] // CIRCLE_FRAMES
    turn_direction = 1 if circle_number % 2 == 0 else -1
    return SEARCH_SPEED, turn_direction * SEARCH_STEERING


def decide(apple, memory):
    """Return a (speed, steering) command based on the detected apple, and update the memory."""
    if memory["state"] in ("reached", "picking", "holding"):
        return 0.0, 0.0

    if memory["state"] == "backup":
        memory["backup_frames"] -= 1
        if memory["backup_frames"] <= 0:
            memory["state"] = "approach"
        # Steering works the other way in reverse: if the apple is on the right, reversing with the
        # wheels turned LEFT swings the car's nose to the right — toward the apple
        steering = clamp(STEER_GAIN * memory["last_error"], -MAX_STEERING, MAX_STEERING)
        return -BACKUP_SPEED, steering

    if apple is None:
        memory["lost_frames"] += 1
        if memory["state"] == "approach" and memory["lost_frames"] < LOST_PATIENCE:
            # The apple vanished for a moment — keep going slowly in the last direction
            return MIN_APPROACH_SPEED, memory["last_steering"]
        memory["state"] = "search"
        return search(memory)

    memory["lost_frames"] = 0
    memory["search_frames"] = 0
    memory["last_apple"] = apple
    cx, cy, area, _ = apple
    # P-controller: error = how far the apple is from the image center (-1 = left edge, +1 = right edge)
    error = (cx - IMAGE_WIDTH / 2) / (IMAGE_WIDTH / 2)
    memory["last_error"] = error

    if area >= STOP_AREA:
        if abs(error) <= CENTER_TOLERANCE:
            memory["state"] = "reached"
            return 0.0, 0.0
        memory["state"] = "backup"
        memory["backup_frames"] = BACKUP_FRAMES
        return 0.0, 0.0

    memory["state"] = "approach"
    # Apple on the right (error > 0) → steer right (negative)
    steering = clamp(-STEER_GAIN * error, -MAX_STEERING, MAX_STEERING)
    # Slow down as the apple gets closer (as its area grows)
    speed = max(MIN_APPROACH_SPEED, APPROACH_SPEED * (1 - area / STOP_AREA))
    memory["last_steering"] = steering
    return speed, steering


def drive(speed, steering):
    """Motor command: both rear wheels at the same speed, front wheels steered."""
    data.actuator("rear_left_motor").ctrl = speed
    data.actuator("rear_right_motor").ctrl = speed
    data.actuator("steering").ctrl = steering


def move_arm(pose, gripper):
    for name, angle in zip(ARM_JOINTS, pose):
        data.actuator(name).ctrl = angle
    data.actuator("gripper").ctrl = gripper


def arm_pose():
    return [data.actuator(name).ctrl[0] for name in ARM_JOINTS]


def start_picking(memory):
    """Compute the apple's 3D position from its pixel position and start the pick-up motion."""
    cx, cy, _, _ = memory["last_apple"]
    target = pixel_to_chassis(model, data, "front_cam", cx, cy, IMAGE_WIDTH, IMAGE_HEIGHT,
                              ground_z=APPLE_RADIUS)
    try:
        pick = PickSequence(target, arm_pose(), data.actuator("gripper").ctrl[0])
    except ValueError:
        # Out of reach — back up and approach again
        memory["state"] = "backup"
        memory["backup_frames"] = BACKUP_FRAMES
        return None
    memory["state"] = "picking"
    return pick


def camera_image(camera):
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def place_apple(x, y):
    """Put the apple at (x, y) and stop it from moving."""
    data.qpos[APPLE_QPOS:APPLE_QPOS + 3] = [x, y, APPLE_RADIUS]
    data.qpos[APPLE_QPOS + 3:APPLE_QPOS + 7] = [1, 0, 0, 0]
    data.qvel[APPLE_QVEL:APPLE_QVEL + 6] = 0
    mujoco.mj_forward(model, data)


def place_apple_randomly(rng):
    """Put the apple at a random spot 0.5-1.5 m from the car, in any direction."""
    car_x, car_y = data.body("chassis").xpos[:2]
    while True:
        distance = rng.uniform(0.5, 1.5)
        angle = rng.uniform(-np.pi, np.pi)
        x, y = car_x + distance * np.cos(angle), car_y + distance * np.sin(angle)
        if abs(x) < 2.5 and abs(y) < 2.5 and not (1.8 < x < 2.2 and abs(y) < 0.45):  # on the floor, away from the obstacle
            place_apple(x, y)
            return


def draw_state(image_bgr, memory, speed, steering):
    colors = {"search": (0, 200, 255), "approach": (255, 200, 0), "backup": (255, 0, 255),
              "reached": (0, 255, 0), "picking": (0, 255, 0), "holding": (0, 255, 0)}
    cv2.putText(image_bgr, f"{memory['state'].upper()}  speed={speed:.1f} steer={steering:+.2f}",
                (10, IMAGE_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[memory["state"]], 2)


def step(memory, pick):
    """One frame: see → decide → act → simulate. Returns the updated (pick, apple, speed, steering, image)."""
    front = camera_image("front_cam")
    apple, _ = find_apple(front)
    speed, steering = decide(apple, memory)
    drive(speed, steering)

    if memory["state"] == "reached":
        pick = start_picking(memory)
    if pick is not None:
        pose, gripper = pick.update(FRAME_TIME)
        move_arm(pose, gripper)
        if pick.finished:
            memory["state"] = "holding"

    for _ in range(STEPS_PER_FRAME):
        mujoco.mj_step(model, data)
    return pick, apple, speed, steering, front


def reset_arm():
    move_arm([0.0, 0.0, 0.0, 0.0], GRIPPER_CLOSED)


def main():
    rng = np.random.default_rng()
    memory, pick = new_memory(), None
    for _ in range(300):              # let the car settle on the ground
        mujoco.mj_step(model, data)

    while True:
        pick, apple, speed, steering, front = step(memory, pick)

        draw_apple(front, apple)
        draw_state(front, memory, speed, steering)
        cv2.imshow("front_cam", front)
        cv2.imshow("chase_cam", camera_image("chase_cam"))

        key = cv2.waitKey(1)
        if key == ord("r"):
            reset_arm()
            place_apple_randomly(rng)
            memory, pick = new_memory(), None
        if key == ord("q") or cv2.getWindowProperty("front_cam", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
