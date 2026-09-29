"""
Live video from the front camera + red apple detection.
The car drives slowly in a circle and the apple is detected in every frame.

Run:   python camera_view.py
Quit:  click the window and press "q", or close the window
"""
import cv2
import mujoco

from vision import draw_apple, find_apple

model = mujoco.MjModel.from_xml_path("picar.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

# Rendering options: don't draw the sonar's yellow ray
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0


def main():
    while True:
        # Drive slowly and steer left — the car draws a circle
        data.actuator("rear_left_motor").ctrl = 5
        data.actuator("rear_right_motor").ctrl = 5
        data.actuator("steering").ctrl = 0.3

        # 16 steps * 0.002 s = 0.032 s of physics → about 30 frames per second
        for _ in range(16):
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera="front_cam", scene_option=scene_option)
        image_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

        apple, mask = find_apple(image_bgr)
        draw_apple(image_bgr, apple)

        cv2.imshow("front_cam", image_bgr)
        cv2.imshow("mask", mask)

        # Quit on "q" or when the window is closed
        key = cv2.waitKey(1)
        if key == ord("q") or cv2.getWindowProperty("front_cam", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
