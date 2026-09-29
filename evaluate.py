"""
Benchmark the apple seeker: put the apple at random spots and count successful pick-ups.

Run:   python evaluate.py              (20 episodes)
       python evaluate.py 50 --seed 7  (50 episodes, fixed random seed)

An episode succeeds when the robot ends in the "holding" state with the apple more than 10 cm above
the ground two seconds after the pick-up finished. No windows are opened — everything runs headless.
"""
import argparse
import time

import mujoco
import numpy as np

import apple_seeker as seeker
from arm import pixel_to_chassis

TIMEOUT = 70.0  # seconds of simulated time per episode


def apple_in_chassis_frame():
    rotation = seeker.data.body("chassis").xmat.reshape(3, 3)
    apple = seeker.data.qpos[seeker.APPLE_QPOS:seeker.APPLE_QPOS + 3]
    return rotation.T @ (apple - seeker.data.body("chassis").xpos)


def run_episode(rng):
    """Returns (success, seconds, error of the camera-based apple position estimate in meters)."""
    mujoco.mj_resetData(seeker.model, seeker.data)
    seeker.reset_arm()
    for _ in range(300):
        mujoco.mj_step(seeker.model, seeker.data)
    seeker.place_apple_randomly(rng)

    memory, pick = seeker.new_memory(), None
    start_time = seeker.data.time
    estimate_error = None
    while seeker.data.time - start_time < TIMEOUT:
        state_before = memory["state"]
        pick, *_ = seeker.step(memory, pick)
        if state_before != "picking" and memory["state"] == "picking":
            # Compare the camera estimate with the true apple position
            cx, cy, _, _ = memory["last_apple"]
            estimate = pixel_to_chassis(seeker.model, seeker.data, "front_cam", cx, cy,
                                        seeker.IMAGE_WIDTH, seeker.IMAGE_HEIGHT, seeker.APPLE_RADIUS)
            estimate_error = np.linalg.norm(estimate[:2] - apple_in_chassis_frame()[:2])
        if memory["state"] == "holding":
            for _ in range(60):  # hold it for ~2 s
                pick, *_ = seeker.step(memory, pick)
            break

    apple_height = seeker.data.qpos[seeker.APPLE_QPOS + 2]
    success = memory["state"] == "holding" and apple_height > 0.1
    return success, seeker.data.time - start_time, estimate_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, nargs="?", default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results = []
    wall_start = time.time()
    for episode in range(args.episodes):
        success, seconds, error = run_episode(rng)
        results.append((success, seconds, error))
        error_text = "-" if error is None else f"{error * 1000:.1f} mm"
        print(f"{episode:3d}  {'OK  ' if success else 'FAIL'}  {seconds:5.1f} s  position estimate error {error_text}")

    successes = [r for r in results if r[0]]
    errors = [r[2] for r in results if r[2] is not None]
    print(f"\nSuccess: {len(successes)}/{len(results)}")
    if successes:
        print(f"Average time to pick up: {np.mean([r[1] for r in successes]):.1f} s (simulated)")
    if errors:
        print(f"Camera position estimate error: mean {np.mean(errors) * 1000:.1f} mm, max {np.max(errors) * 1000:.1f} mm")
    print(f"Wall-clock time: {time.time() - wall_start:.0f} s")


if __name__ == "__main__":
    main()
