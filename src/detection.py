import cv2
import numpy as np
from src.utils import *

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

    image_cropped = image[max_y:max_y+max_h, max_x:max_x+max_w]
    cv2.rectangle(image_res, (max_x, max_y), (max_x + max_w, max_y + max_h), color, 2)
    return image_res, image_cropped, (max_x, max_y, max_w, max_h)


def get_playing_fields(image: np.ndarray, rect_size: tuple[int], color: tuple[int] = (0, 0, 255), display_steps: bool = False) -> tuple[np.ndarray, list[int]]:

    tiles_coords = list()
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
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w < rect_size[0] or w > rect_size[1] or h < rect_size[0] or h > rect_size[1] or w/h > 1.4 or h/w > 1.4:
            continue
        cv2.rectangle(image_res, (x, y), (x + w, y + h), color, 2)
        tiles_coords.append(((x, y), (x + w, y + h)))

    if display_steps:
        imshow(np.concatenate([gray, thresh], 1))
        imshow(cv2.resize(image_res, None, fx=0.8, fy=0.8))
    return image_res, tiles_coords


def get_counters_coords(counters_img:np.ndarray, fields_coords:list[np.ndarray], display:bool=False) -> tuple[tuple[int], float]:
    frame = counters_img.copy()
    frame = cv2.medianBlur(frame, 5)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if display:
        imshow(np.concatenate([frame_gray, bin_mask], 1))

    coords_ranked = list()
    for p1, p2 in fields_coords:
        img_cropped_bin = bin_mask[p1[1]:p2[1], p1[0]:p2[0]]
        # imshow(img_cropped_bin)
        white_percentage = get_percentage_white_pixels(img_cropped_bin)
        coords_ranked.append(((p1, p2), white_percentage))

    coords_ranked.sort(key = lambda x: x[1], reverse=True)
    return coords_ranked


def check_intersection(boxA: tuple, boxB: tuple) -> bool:
    x = max(boxA[0], boxB[0])
    y = max(boxA[1], boxB[1])
    w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
    h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y

    intersected = True
    if w < 0 or h < 0:
        intersected = False

    return intersected


def has_won(checkers: list[tuple]) -> bool:
    """
    :param checkers: list of all checkers of one color
    """
    xs = [item[0] for item in checkers]
    ys = [item[1] for item in checkers]
    return np.allclose(xs, xs[0], atol=5) or np.allclose(ys, ys[1], atol=5)
