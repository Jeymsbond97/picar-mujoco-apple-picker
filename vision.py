"""
Computer vision helpers: find the red apple in a camera image.
Used by camera_view.py and apple_seeker.py.
"""
import cv2
import numpy as np

# Shape thresholds that separate the apple from a red box and other red things
MIN_AREA = 20            # smaller blobs are noise
MIN_CIRCULARITY = 0.80   # 4πA/P²: circle = 1.0, square ≈ 0.785. Measured: apple 0.82-0.91, box ≤ 0.79
MIN_CORNERS = 5          # after polygon simplification a box always has 4 corners, the apple 5-8


def red_mask(image_bgr):
    """Binary image: red pixels are white (255), everything else black (0).
    Red sits at both ends of the hue circle (0-10 and 170-180), hence two ranges."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    return mask1 | mask2


def looks_like_apple(contour):
    """Does the blob look like an apple: large enough, round, and not a rectangle."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area < MIN_AREA or perimeter == 0:
        return False
    circularity = 4 * np.pi * area / perimeter ** 2
    corners = len(cv2.approxPolyDP(contour, 0.04 * perimeter, True))
    return circularity >= MIN_CIRCULARITY and corners >= MIN_CORNERS


def find_apple(image_bgr):
    """Look for the apple. Returns (cx, cy, area, box) or None, plus the mask."""
    mask = red_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if looks_like_apple(c)]
    if not candidates:
        return None, mask
    apple = max(candidates, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(apple)
    return (x + w // 2, y + h // 2, cv2.contourArea(apple), (x, y, w, h)), mask


def draw_apple(image_bgr, apple):
    """Draw the detected apple (green box + center) or write "no apple"."""
    if apple is None:
        cv2.putText(image_bgr, "no apple", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return
    cx, cy, area, (x, y, w, h) = apple
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(image_bgr, (cx, cy), 4, (0, 255, 0), -1)
    cv2.putText(image_bgr, f"apple: cx={cx} cy={cy} area={area:.0f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
