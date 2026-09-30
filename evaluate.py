"""
Benchmark the apple seeker: random apple + random obstacles, count clean pick-ups.

Run:   python evaluate.py              (20 episodes)
       python evaluate.py 50 --seed 7  (50 episodes, fixed random seed)

An episode succeeds when the robot ends in the "holding" state with the apple more than 10 cm above
the ground two seconds after the pick-up finished, WITHOUT ever touching an obstacle.
No windows are opened — everything runs headless.
"""
import argparse
import time

import mujoco
import numpy as np

import apple_seeker as seeker
from arm import pixel_to_chassis

TIMEOUT = 90.0  # seconds of simulated time per episode

ROBOT_GEOMS = {i for i in range(seeker.model.ngeom)
               if seeker.model.body_rootid[seeker.model.geom_bodyid[i]] == seeker.model.body("chassis").id}
OBSTACLE_GEOMS = {seeker.model.geom(name).id for name in seeker.OBSTACLES + ("obstacle",)}


def touching_obstacle():
    for contact in seeker.data.contact[:seeker.data.ncon]:
        pair = {contact.geom1, contact.geom2}
        if pair & ROBOT_GEOMS and pair & OBSTACLE_GEOMS:
            return True
    return False


def apple_in_chassis_frame():
    rotation = seeker.data.body("chassis").xmat.reshape(3, 3)
    apple = seeker.data.qpos[seeker.APPLE_QPOS:seeker.APPLE_QPOS + 3]
    return rotation.T @ (apple - seeker.data.body("chassis").xpos)


def run_episode(rng):
    """Returns (picked_up, touches, seconds, evades, position estimate error in meters)."""
    mujoco.mj_resetData(seeker.model, seeker.data)
    for _ in range(300):
        mujoco.mj_step(seeker.model, seeker.data)
    seeker.new_scene(rng)

    memory, pick = seeker.new_memory(), None
    start_time = seeker.data.time
    estimate_error = None
    touches, evades, was_touching = 0, 0, False
    while seeker.data.time - start_time < TIMEOUT:
        state_before = memory["state"]
        pick, *_ = seeker.step(memory, pick)
        touching = touching_obstacle()
        touches += touching and not was_touching           # count each new touch once
        was_touching = touching
        evades += memory["state"] == "evade" and state_before != "evade"
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

    picked_up = memory["state"] == "holding" and seeker.data.qpos[seeker.APPLE_QPOS + 2] > 0.1
    return picked_up, touches, seeker.data.time - start_time, evades, estimate_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, nargs="?", default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results = []
    wall_start = time.time()
    for episode in range(args.episodes):
        picked_up, touches, seconds, evades, error = run_episode(rng)
        success = picked_up and touches == 0
        results.append((success, picked_up, touches, seconds, error))
        error_text = "-" if error is None else f"{error * 1000:.1f} mm"
        print(f"{episode:3d}  {'OK  ' if success else 'FAIL'}  picked={'yes' if picked_up else 'no '}  "
              f"touches={touches}  evades={evades}  {seconds:5.1f} s  estimate error {error_text}")

    successes = [r for r in results if r[0]]
    errors = [r[4] for r in results if r[4] is not None]
    print(f"\nSuccess (picked up, no obstacle touched): {len(successes)}/{len(results)}")
    print(f"Picked up: {sum(r[1] for r in results)}/{len(results)}   "
          f"episodes with a touch: {sum(r[2] > 0 for r in results)}/{len(results)}")
    if successes:
        print(f"Average time to pick up: {np.mean([r[3] for r in successes]):.1f} s (simulated)")
    if errors:
        print(f"Camera position estimate error: mean {np.mean(errors) * 1000:.1f} mm, max {np.max(errors) * 1000:.1f} mm")
    print(f"Wall-clock time: {time.time() - wall_start:.0f} s")


if __name__ == "__main__":
    main()
