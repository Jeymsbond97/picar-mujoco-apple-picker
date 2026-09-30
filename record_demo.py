"""
Record a demo video straight from the simulation (no screen capture needed).

Left half:  the robot's front camera with the detection overlay and the current state.
Right half: a third-person camera that follows the car.

Run:   python record_demo.py                  → media/demo_obstacles.mp4 (3 episodes)
       python record_demo.py --episodes 5 --seed 11
"""
import argparse

import cv2
import mujoco
import numpy as np

import apple_seeker as seeker
from vision import draw_apple

FPS = round(1 / seeker.FRAME_TIME)   # one video frame per brain step → real-time playback
HOLD_SECONDS = 2.0                   # keep filming this long after the apple is lifted
TIMEOUT = 70.0                       # give up on an episode after this much simulated time

side_renderer = mujoco.Renderer(seeker.model, height=seeker.IMAGE_HEIGHT, width=seeker.IMAGE_WIDTH)
side_camera = mujoco.MjvCamera()
side_camera.distance = 0.9
side_camera.azimuth = 135
side_camera.elevation = -20


def side_view():
    """Third-person view that follows the car (smoothly, without rotating with it)."""
    side_camera.lookat[:] = seeker.data.body("chassis").xpos + [0, 0, 0.08]
    side_renderer.update_scene(seeker.data, camera=side_camera, scene_option=seeker.scene_option)
    return cv2.cvtColor(side_renderer.render(), cv2.COLOR_RGB2BGR)


def label(image, text):
    cv2.putText(image, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def record_episode(writer, rng, number, total):
    mujoco.mj_resetData(seeker.model, seeker.data)
    for _ in range(300):
        mujoco.mj_step(seeker.model, seeker.data)
    seeker.new_scene(rng)   # random apple + random obstacles

    memory, pick = seeker.new_memory(), None
    start = seeker.data.time
    held_since = None
    while seeker.data.time - start < TIMEOUT:
        pick, apple, speed, steering, front = seeker.step(memory, pick)
        draw_apple(front, apple)
        seeker.draw_state(front, memory, speed, steering)
        label(front, f"episode {number}/{total}   t = {seeker.data.time - start:4.1f} s")
        writer.write(np.concatenate([front, side_view()], axis=1))

        if memory["state"] == "holding":
            held_since = held_since or seeker.data.time
            if seeker.data.time - held_since > HOLD_SECONDS:
                break
    return memory["state"] == "holding"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output", default="media/demo_obstacles.mp4")
    args = parser.parse_args()

    size = (2 * seeker.IMAGE_WIDTH, seeker.IMAGE_HEIGHT)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"avc1"), FPS, size)  # avc1 = H.264
    rng = np.random.default_rng(args.seed)
    for number in range(1, args.episodes + 1):
        success = record_episode(writer, rng, number, args.episodes)
        print(f"episode {number}: {'picked up' if success else 'failed'}")
    writer.release()
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
