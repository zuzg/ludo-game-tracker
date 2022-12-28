import cv2
import numpy as np


def detect_board(image: np.ndarray, color: tuple[int] = (0, 0, 255)) -> tuple[np.ndarray, list[int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # reduce image noise
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.medianBlur(gray, 7)
    # detect the edges
    edges = cv2.Canny(gray, 50, 150)
    # morphological operations applied to refine the edges of the fields
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(edges, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    # contour detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # draw contours on the original image
    image_res = image.copy()
    max_w = 0
    max_h = 0
    max_x, max_y = None, None
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if w * h > max_w * max_h: 
            max_w = w
            max_h = h
            max_x, max_y = x, y

    cv2.rectangle(image_res, (max_x, max_y), (max_x + max_w, max_y + max_h), color, 2)
    return image_res, (max_x, max_y, max_w, max_h)


def check_intersection(boxA: tuple, boxB: tuple) -> bool:
    x = max(boxA[0], boxB[0])
    y = max(boxA[1], boxB[1])
    w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
    h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y

    intersected = True
    if w < 0 or h < 0:
        intersected = False

    return intersected
