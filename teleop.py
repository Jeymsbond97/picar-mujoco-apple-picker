"""
Drive the PiCar with the keyboard (teleoperation).
HOLD a key to keep moving, RELEASE it to stop.

Run (on macOS only with mjpython!):
    source venv/bin/activate
    mjpython teleop.py

Keys (hold them):
    Up / Down arrow      drive forward / backward (slows to a stop when released)
    Left / Right arrow   steer left / right (re-centers when released)
    Enter                emergency stop
    Z / X                turn the arm left / right (arm_yaw)
    U / J                shoulder forward / back
    Y / H                elbow forward / back
    V / B                wrist down / up
    O / C                open / close the gripper
The arm and gripper stay where they are when the key is released.
"""
import ctypes
import ctypes.util
import time

import mujoco
import mujoco.viewer

# --- Read the keyboard state directly from macOS ---
# The MuJoCo viewer never reports key releases, so every frame we ask macOS
# "is this key pressed RIGHT NOW?" (CGEventSourceKeyState, called through ctypes).
_quartz = ctypes.cdll.LoadLibrary(ctypes.util.find_library("ApplicationServices"))
_quartz.CGEventSourceKeyState.argtypes = [ctypes.c_int32, ctypes.c_uint16]
_quartz.CGEventSourceKeyState.restype = ctypes.c_bool
_quartz.CGPreflightListenEventAccess.restype = ctypes.c_bool
HID_SYSTEM_STATE = 1  # the physical keyboard state


def is_pressed(key):
    """Is the key (macOS virtual key code) held down right now?"""
    return _quartz.CGEventSourceKeyState(HID_SYSTEM_STATE, key)


# macOS virtual key codes. They are NOT the ASCII codes of the letters.
KEY_RETURN = 36
KEY_LEFT = 123
KEY_RIGHT = 124
KEY_DOWN = 125
KEY_UP = 126
KEY_Z, KEY_X = 6, 7
KEY_U, KEY_J = 32, 38
KEY_Y, KEY_H = 16, 4
KEY_V, KEY_B = 9, 11
KEY_O, KEY_C = 31, 8

# Motion rates (per second)
MAX_SPEED = 12.0    # wheel speed, rad/s (12 * 0.045 m = ~0.54 m/s)
ACCELERATION = 20.0 # how fast the speed changes (smooth acceleration and braking)
STEER_RATE = 1.5    # steering rate, rad/s
ARM_RATE = 0.6      # arm joint rate, rad/s
GRIP_RATE = 0.02    # finger opening rate, m/s

# Joint -> (key for the positive direction, key for the negative direction, rate)
JOINT_KEYS = {
    "arm_yaw":  (KEY_Z, KEY_X, ARM_RATE),
    "shoulder": (KEY_U, KEY_J, ARM_RATE),
    "elbow":    (KEY_Y, KEY_H, ARM_RATE),
    "wrist":    (KEY_V, KEY_B, ARM_RATE),
    "gripper":  (KEY_O, KEY_C, GRIP_RATE),
}
ALL_KEYS = [KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT]
for positive_key, negative_key, _ in JOINT_KEYS.values():
    ALL_KEYS += [positive_key, negative_key]

model = mujoco.MjModel.from_xml_path("picar.xml")
data = mujoco.MjData(model)

# Current commands. The main loop updates them smoothly every frame and sends them to the motors.
command = {"speed": 0.0, "steering": 0.0, "arm_yaw": 0.0, "shoulder": 0.0,
           "elbow": 0.0, "wrist": 0.0, "gripper": 0.0}


def clamp(value, low, high):
    """Keep value within [low, high]."""
    return max(low, min(high, value))


def move_toward(value, target, max_change):
    """Move value toward target by at most max_change (smooth change)."""
    return value + clamp(target - value, -max_change, max_change)


def limits(name):
    """Allowed command range, taken from ctrlrange in the XML."""
    low, high = model.actuator(name).ctrlrange
    return low, high


def direction(positive_key, negative_key):
    """+1 = positive key held, -1 = negative key held, 0 = neither (or both)."""
    return int(is_pressed(positive_key)) - int(is_pressed(negative_key))


def update(dt):
    """Smoothly update the commands for dt seconds based on the keys being held."""
    if is_pressed(KEY_RETURN):
        command["speed"] = 0.0
        command["steering"] = 0.0
        return

    target_speed = MAX_SPEED * direction(KEY_UP, KEY_DOWN)
    command["speed"] = move_toward(command["speed"], target_speed, ACCELERATION * dt)

    max_steer = limits("steering")[1]
    target_steer = max_steer * direction(KEY_LEFT, KEY_RIGHT)
    command["steering"] = move_toward(command["steering"], target_steer, STEER_RATE * dt)

    for name, (positive_key, negative_key, rate) in JOINT_KEYS.items():
        step = direction(positive_key, negative_key) * rate * dt
        command[name] = clamp(command[name] + step, *limits(name))


def apply_command():
    """Write the commands to the motors (data.ctrl)."""
    data.actuator("rear_left_motor").ctrl = command["speed"]
    data.actuator("rear_right_motor").ctrl = command["speed"]
    for name in ("steering", "arm_yaw", "shoulder", "elbow", "wrist", "gripper"):
        data.actuator(name).ctrl = command[name]


def print_status():
    print(f"speed={command['speed']:+5.1f}  steer={command['steering']:+.2f}  "
          f"yaw={command['arm_yaw']:+.2f} shoulder={command['shoulder']:+.2f} "
          f"elbow={command['elbow']:+.2f} wrist={command['wrist']:+.2f} "
          f"grip={command['gripper']:.3f}")


def main():
    if not _quartz.CGPreflightListenEventAccess():
        print("⚠️  No permission to read the keyboard. Enable your terminal (Terminal or VS Code) under\n"
              "    System Settings → Privacy & Security → Input Monitoring,\n"
              "    then reopen the terminal and run the program again.")

    frame_time = 1 / 60                                     # redraw the screen 60 times per second
    steps_per_frame = int(frame_time / model.opt.timestep)  # physics steps per frame
    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # The viewer toggles visualization flags (joints, contacts...) on letter keys.
        # Remember their initial state and restore it every frame so the display doesn't change.
        saved_flags = viewer.opt.flags.copy()

        while viewer.is_running():
            frame_start = time.monotonic()

            update(frame_time)
            apply_command()
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            with viewer.lock():
                viewer.opt.flags[:] = saved_flags
            viewer.sync()

            # While a key is held, print the status 4 times per second
            if any(is_pressed(key) for key in ALL_KEYS) and frame_start - last_print > 0.25:
                print_status()
                last_print = frame_start

            # Wait until the end of the frame so the simulation doesn't run faster than real time
            time_left = frame_time - (time.monotonic() - frame_start)
            if time_left > 0:
                time.sleep(time_left)


if __name__ == "__main__":
    main()
