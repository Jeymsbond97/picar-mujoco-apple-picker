"""
The robot's "brain": finds the red apple with the front camera, drives to it while steering around
obstacles, and picks it up with the arm.

States (state machine):
    search   — no apple in view: drive a figure-eight — one full circle, then one full circle the other way.
               Circling in only one direction never sees an apple sitting inside the turning circle
               (the camera always faces outward along the circle); on the opposite circle it lies outside.
               The first circle turns toward the side where the apple was last seen.
    explore  — a whole figure-eight found nothing (the apple is probably hidden behind obstacles):
               drive straight for a few seconds and search again from a new spot
    approach — apple in view: steer toward it (P-controller) and drive closer
    goto     — apple not in view any more (e.g. while driving around an obstacle) but we remember where it
               is: drive toward the remembered position
    backup   — apple is close but off-center (out of the arm's reach): back up a bit and re-align
    evade    — an obstacle is right in front, or the car is stuck: reverse while turning toward the open side
    reached  — apple is close and centered: stop and compute its 3D position from the camera
    picking  — the arm is grasping and lifting the apple
    holding  — the apple is in the gripper

While driving, the steering is the sum of two "forces" (a potential field):
the pull toward the goal and a push away from every obstacle the range sensors see.

Simplification: to remember the apple's position the robot needs to know its own pose. Here the true pose
from the simulator is used as "odometry"; a real robot would estimate it from wheel encoders and an IMU,
and that estimate drifts over time.

Run:   python apple_seeker.py
Keys (click a window first):  r = new random apple and obstacles,  q = quit
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
CLOSE_AREA = 1500        # apple bigger than this (~0.45 m away) is "close"...
CLOSE_MAX_ERROR = 0.6    # ...and if it is also this far off-center, the car can't turn onto it: back up instead
                         # of circling around it (circling this close clips the apple and pushes it away)
BACKUP_SPEED = 5.0       # reverse speed while re-aligning
BACKUP_FRAMES = 40       # ~1.3 s of reversing
LOST_PATIENCE = 10       # frames without the apple before we call it lost (tolerates one-frame misses)
APPLE_RADIUS = 0.02
EXPLORE_FRAMES = 90      # after a whole figure-eight without the apple: drive straight ~3 s to a new spot
GOTO_SPEED = 7.0         # speed toward the remembered apple position
GOTO_GAIN = 1.5          # steering per radian of bearing error toward the remembered position
GOTO_ARRIVE = 0.3        # if we are this close to the remembered position and still don't see it → forget it

# Obstacle avoidance with range sensors: a sonar fan at the bumper plus two corner sensors
RANGE_SENSORS = ("sonar_left_30", "sonar_left_15", "sonar", "sonar_right_15", "sonar_right_30",
                 "corner_left", "corner_right", "side_left", "side_right")
SIDE_CLEARANCE = 0.1     # an obstacle this close beside the car → don't turn toward that side
SONAR_MAX = 2.0          # a rangefinder reports -1 when nothing is within this range
BUMPER_X = 0.14          # front of the car in the chassis frame
CORRIDOR_HALF_WIDTH = 0.16  # obstacles closer than this to the car's centerline are "in the path" (car is 0.24 m wide)
INFLUENCE_DISTANCE = 0.5    # obstacles farther than this don't push the steering
AVOID_GAIN = 1.2            # strength of the push away from obstacles
EVADE_DISTANCE = 0.12       # an obstacle in the path closer than this → stop and evade
CORNER_DANGER = 0.06        # something this close to a front corner → stop and evade
EVADE_SPEED = 5.0
EVADE_FRAMES = 35           # ~1.1 s of reversing
STUCK_FRAMES = 45           # if the car barely moved in this many frames (~1.4 s) while driving → stuck
STUCK_DISTANCE = 0.02

OBSTACLES = ("block_1", "block_2", "block_3", "pillar_1", "pillar_2", "block_4")
CAR_CLEARANCE = 0.35     # free space around the car's start position
APPLE_CLEARANCE = 0.3    # free space around the apple (so it can be approached and grasped)
GAP = 0.3                # minimum gap between obstacles (wider than the car)

model = mujoco.MjModel.from_xml_path("picar.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
MAX_STEERING = model.actuator("steering").ctrlrange[1]
ARM_JOINTS = ("arm_yaw", "shoulder", "elbow", "wrist")

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

APPLE_QPOS = model.jnt_qposadr[model.body("apple").jntadr[0]]  # where the apple's freejoint lives in qpos
APPLE_QVEL = model.jnt_dofadr[model.body("apple").jntadr[0]]


def sensor_ray(name):
    """Where a rangefinder's ray starts (x, y) and its angle in degrees (+ = left), read from the XML."""
    site = model.site(model.sensor(name).objid[0])
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, site.quat)
    direction = rotation.reshape(3, 3)[:, 2]           # a rangefinder measures along the site's z axis
    return site.pos[0], site.pos[1], np.degrees(np.arctan2(direction[1], direction[0]))


RAYS = {name: sensor_ray(name) for name in RANGE_SENSORS}


def clamp(value, low, high):
    return max(low, min(high, value))


def new_memory():
    """The brain's memory."""
    return {"state": "search", "lost_frames": 0, "last_steering": 0.0, "last_error": 0.0,
            "search_frames": 0, "search_direction": 1, "unseen_frames": 0, "backup_frames": 0,
            "last_apple": None, "apple_world": None,
            "evade_frames": 0, "evade_steering": 0.0, "positions": []}


def chassis_to_world(point):
    rotation = data.body("chassis").xmat.reshape(3, 3)
    return rotation @ point + data.body("chassis").xpos


def world_to_chassis(point):
    rotation = data.body("chassis").xmat.reshape(3, 3)
    return rotation.T @ (point - data.body("chassis").xpos)


# ---------------------------------------------------------------- obstacle sensing and avoidance

def read_sonars():
    """Obstacle points seen by the range sensors, as a list of (angle_deg, distance, ahead_x, sideways_y).
    ahead_x is measured from the front bumper, sideways_y from the car's centerline."""
    points = []
    for name, (origin_x, origin_y, angle) in RAYS.items():
        distance = data.sensor(name).data[0]
        if distance < 0:
            distance = SONAR_MAX
        radians = np.deg2rad(angle)
        ahead = origin_x + distance * np.cos(radians) - BUMPER_X
        sideways = origin_y + distance * np.sin(radians)
        points.append((angle, distance, ahead, sideways))
    return points


def path_clearance(points):
    """How far the car can drive straight before hitting something in its path."""
    in_path = [x for _, distance, x, y in points if abs(y) < CORRIDOR_HALF_WIDTH and distance < SONAR_MAX]
    return min(in_path, default=SONAR_MAX)


def corner_blocked(points):
    """Something is right beside a front wheel."""
    return any(abs(angle) > 45 and distance < CORNER_DANGER for angle, distance, _, _ in points)


def open_side(points):
    """+1 if the left is safer, -1 if the right is. A side is judged by its CLOSEST obstacle: summing the
    distances made one obstacle 6 cm away look harmless when the other rays on that side saw nothing."""
    left = min(distance for angle, distance, _, _ in points if angle > 0)
    right = min(distance for angle, distance, _, _ in points if angle < 0)
    return 1 if left >= right else -1


def avoidance(points):
    """Steering push away from nearby obstacles and how close the closest one is (0 = far, 1 = touching).
    An obstacle on the left pushes the steering right, and the other way round; one straight ahead pushes
    toward the more open side."""
    push, nearest = 0.0, 0.0
    for angle, distance, _, _ in points:
        if distance >= INFLUENCE_DISTANCE or abs(angle) > 75:   # side sensors are handled by side_limit()
            continue
        weight = (INFLUENCE_DISTANCE - distance) / INFLUENCE_DISTANCE
        away = -np.sign(angle) if angle != 0 else open_side(points)
        push += away * weight
        nearest = max(nearest, weight)
    return AVOID_GAIN * push, nearest


def is_stuck(memory, speed):
    """The car is commanded to move but has barely moved lately (e.g. wedged against something)."""
    memory["positions"].append(data.body("chassis").xpos[:2].copy())
    memory["positions"] = memory["positions"][-STUCK_FRAMES:]
    if abs(speed) < 1.0 or len(memory["positions"]) < STUCK_FRAMES:
        return False
    return np.linalg.norm(memory["positions"][-1] - memory["positions"][0]) < STUCK_DISTANCE


def start_evade(memory, points):
    # Reversing with the wheels turned right swings the nose LEFT — so steer opposite to the open side
    memory["state"] = "evade"
    memory["evade_frames"] = EVADE_FRAMES
    memory["evade_steering"] = -open_side(points) * MAX_STEERING
    memory["positions"] = []
    # Afterwards search by circling toward the open side — circling the old way leads straight back into it
    memory["search_direction"] = open_side(points)
    memory["search_frames"] = 0


def side_limit(steering, points):
    """Don't turn toward a side where an obstacle is right beside the car (the inner rear wheel would clip it)."""
    for angle, distance, _, _ in points:
        if abs(angle) > 75 and distance < SIDE_CLEARANCE:
            if angle > 0:
                steering = min(steering, 0.0)   # something on the left → no left turn
            else:
                steering = max(steering, 0.0)   # something on the right → no right turn
    return steering


def with_avoidance(speed, steering, points):
    """Add the obstacle push to a (speed, steering) command and slow down near obstacles."""
    push, nearest = avoidance(points)
    steering = clamp((1 - nearest) * steering + push, -MAX_STEERING, MAX_STEERING)
    steering = side_limit(steering, points)
    speed *= clamp(path_clearance(points) / INFLUENCE_DISTANCE, 0.4, 1.0)
    return speed, steering


# ---------------------------------------------------------------- decision making

def search(memory):
    """Search pattern: figure-eight — CIRCLE_FRAMES frames turning one way, then as many turning the other way.
    If a whole figure-eight finds nothing, explore: drive straight for EXPLORE_FRAMES, then search again."""
    memory["unseen_frames"] += 1
    if memory["unseen_frames"] > 2 * CIRCLE_FRAMES:
        memory["state"] = "explore"
        if memory["unseen_frames"] > 2 * CIRCLE_FRAMES + EXPLORE_FRAMES:
            memory["unseen_frames"] = 0
            memory["search_frames"] = 0
        return SEARCH_SPEED, 0.0
    memory["state"] = "search"
    memory["search_frames"] += 1
    circle_number = memory["search_frames"] // CIRCLE_FRAMES
    turn_direction = memory["search_direction"] * (1 if circle_number % 2 == 0 else -1)
    return SEARCH_SPEED, turn_direction * SEARCH_STEERING


def go_to_remembered_apple(points, memory):
    """Drive toward where the apple was last seen. Returns None when there is nothing useful to go to."""
    if memory["apple_world"] is None:
        return None
    target = world_to_chassis(memory["apple_world"])
    if np.hypot(target[0], target[1]) < GOTO_ARRIVE:
        memory["apple_world"] = None          # we are there and still don't see it — the memory was wrong
        return None
    memory["state"] = "goto"
    bearing = np.arctan2(target[1], target[0])  # + = the apple is to the left
    steering = clamp(GOTO_GAIN * bearing, -MAX_STEERING, MAX_STEERING)
    return with_avoidance(GOTO_SPEED, steering, points)


def remember_apple(apple, memory):
    """Store the apple's position in world coordinates (from its pixel, via the ground plane)."""
    cx, cy, _, _ = apple
    in_chassis = pixel_to_chassis(model, data, "front_cam", cx, cy, IMAGE_WIDTH, IMAGE_HEIGHT,
                                  ground_z=APPLE_RADIUS)
    memory["apple_world"] = chassis_to_world(in_chassis)


def decide(apple, points, memory):
    """Return a (speed, steering) command from the detected apple and the sonar points, and update the memory."""
    if memory["state"] in ("reached", "picking", "holding"):
        return 0.0, 0.0

    if memory["state"] == "evade":
        memory["evade_frames"] -= 1
        if memory["evade_frames"] <= 0:
            memory["state"] = "search"
        return -EVADE_SPEED, memory["evade_steering"]

    if memory["state"] == "backup":
        memory["backup_frames"] -= 1
        if memory["backup_frames"] <= 0:
            memory["state"] = "approach"
        # Steering works the other way in reverse: if the apple is on the right, reversing with the
        # wheels turned LEFT swings the car's nose to the right — toward the apple
        steering = clamp(STEER_GAIN * memory["last_error"], -MAX_STEERING, MAX_STEERING)
        return -BACKUP_SPEED, steering

    # An obstacle right in front or beside a front wheel: don't wait to hit it
    if path_clearance(points) < EVADE_DISTANCE or corner_blocked(points):
        start_evade(memory, points)
        return 0.0, 0.0

    if apple is None:
        memory["lost_frames"] += 1
        if memory["state"] == "approach" and memory["lost_frames"] < LOST_PATIENCE:
            # The apple vanished for a moment — keep going slowly in the last direction
            return with_avoidance(MIN_APPROACH_SPEED, memory["last_steering"], points)
        command = go_to_remembered_apple(points, memory)
        if command is not None:
            return command
        if memory["state"] not in ("search", "explore"):
            # Just lost it: start searching toward the side where it was last seen
            memory["search_direction"] = -1 if memory["last_error"] > 0 else 1
            memory["search_frames"] = 0
            memory["unseen_frames"] = 0
        return with_avoidance(*search(memory), points)

    memory["lost_frames"] = 0
    memory["search_frames"] = 0
    memory["unseen_frames"] = 0
    memory["last_apple"] = apple
    remember_apple(apple, memory)
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
    if area >= CLOSE_AREA and abs(error) > CLOSE_MAX_ERROR:
        memory["state"] = "backup"
        memory["backup_frames"] = BACKUP_FRAMES
        return 0.0, 0.0

    memory["state"] = "approach"
    # Apple on the right (error > 0) → steer right (negative)
    steering = clamp(-STEER_GAIN * error, -MAX_STEERING, MAX_STEERING)
    # Slow down as the apple gets closer (as its area grows)
    speed = max(MIN_APPROACH_SPEED, APPROACH_SPEED * (1 - area / STOP_AREA))
    memory["last_steering"] = steering
    return with_avoidance(speed, steering, points)


# ---------------------------------------------------------------- acting

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


# ---------------------------------------------------------------- scene setup

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
        if abs(x) < 2.5 and abs(y) < 2.5 and not (1.8 < x < 2.2 and abs(y) < 0.45):  # on the floor, away from the wall
            place_apple(x, y)
            return


def footprint_radius(name):
    """Radius of the circle that covers the obstacle seen from above."""
    geom = model.geom(name)
    if geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.hypot(geom.size[0], geom.size[1]))
    return float(geom.size[0])


def place_obstacles_randomly(rng):
    """Scatter the obstacles around the car and the apple. The first one goes roughly on the straight line
    between them, so the robot has to drive around it. Obstacles that don't fit are parked far away."""
    car = data.body("chassis").xpos[:2].copy()
    apple = data.qpos[APPLE_QPOS:APPLE_QPOS + 2].copy()
    placed = []
    for index, name in enumerate(OBSTACLES):
        radius = footprint_radius(name)
        position = np.array([20.0 + 2 * index, 20.0])      # "parked" if no free spot is found
        for _ in range(300):
            if index == 0:
                t = rng.uniform(0.4, 0.6)
                side = np.array([-(apple - car)[1], (apple - car)[0]]) / np.linalg.norm(apple - car)
                candidate = car + t * (apple - car) + rng.uniform(-0.08, 0.08) * side
            else:
                candidate = car + rng.uniform(-1.6, 1.6, size=2)
            if (np.linalg.norm(candidate - car) > radius + CAR_CLEARANCE
                    and np.linalg.norm(candidate - apple) > radius + APPLE_CLEARANCE
                    and all(np.linalg.norm(candidate - p) > radius + r + GAP for p, r in placed)
                    and not (1.7 < candidate[0] < 2.3 and abs(candidate[1]) < 0.5)):   # not inside the wall
                position = candidate
                placed.append((candidate, radius))
                break
        model.geom(name).pos[:2] = position
        yaw = rng.uniform(0, np.pi)
        model.geom(name).quat[:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)


def park_cubes():
    """Move the small cubes out of the arena. They are for grasping practice in teleop.py, not obstacles:
    at 3 cm tall they are below the sonar, so the car would just bulldoze them."""
    for index, name in enumerate(("blue_cube", "green_cube")):
        address = model.jnt_qposadr[model.body(name).jntadr[0]]
        data.qpos[address:address + 3] = [-20.0 - index, -20.0, 0.015]


def new_scene(rng):
    reset_arm()
    park_cubes()
    place_apple_randomly(rng)
    place_obstacles_randomly(rng)


# ---------------------------------------------------------------- main loop

def draw_state(image_bgr, memory, speed, steering):
    colors = {"search": (0, 200, 255), "approach": (255, 200, 0), "backup": (255, 0, 255),
              "evade": (0, 0, 255), "explore": (0, 140, 255), "goto": (255, 255, 0), "reached": (0, 255, 0), "picking": (0, 255, 0), "holding": (0, 255, 0)}
    cv2.putText(image_bgr, f"{memory['state'].upper()}  speed={speed:.1f} steer={steering:+.2f}",
                (10, IMAGE_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[memory["state"]], 2)


def step(memory, pick):
    """One frame: see → decide → act → simulate. Returns the updated (pick, apple, speed, steering, image)."""
    front = camera_image("front_cam")
    apple, _ = find_apple(front)
    points = read_sonars()
    speed, steering = decide(apple, points, memory)
    if memory["state"] in ("search", "explore", "approach", "goto") and is_stuck(memory, speed):
        start_evade(memory, points)
        speed, steering = 0.0, 0.0
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
    new_scene(rng)

    while True:
        pick, apple, speed, steering, front = step(memory, pick)

        draw_apple(front, apple)
        draw_state(front, memory, speed, steering)
        cv2.imshow("front_cam", front)
        cv2.imshow("chase_cam", camera_image("chase_cam"))

        key = cv2.waitKey(1)
        if key == ord("r"):
            new_scene(rng)
            memory, pick = new_memory(), None
        if key == ord("q") or cv2.getWindowProperty("front_cam", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
