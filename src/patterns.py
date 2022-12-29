import numpy as np
import cv2
from src.utils import *


def match_patterns(board: np.ndarray, templates: list[str], color: tuple[int], threshold: float = 0.8) -> tuple[np.ndarray, list[int]]:
    coords = list()
    board_matched = board.copy()
    board_gray = cv2.cvtColor(board_matched, cv2.COLOR_BGR2GRAY)
    for t in templates:
        # print(t)
        template = cv2.imread(t, 1)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        h, w = template.shape[0], template.shape[1]
        corr = cv2.matchTemplate(board_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(corr >= threshold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(board_matched, pt, (pt[0] + w, pt[1] + h), color, 2)
            coords.append((pt, (pt[0] + w, pt[1] + h)))

    return board_matched, coords


def check_match_template(image: np.ndarray, templates: list[np.ndarray], threshold: float = 0.8) -> bool:
    for template in templates:
        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
        # imshow(image)
        if image.shape[0] < template_gray.shape[0] or image.shape[1] < template_gray.shape[1]:
            if image.shape[1] > template_gray.shape[1] or image.shape[0] > template_gray.shape[0]:
                return False
            image, template_gray = template_gray, image
        res = cv2.matchTemplate(image, template_gray, cv2.TM_CCOEFF_NORMED)
        if np.any(np.where(res > threshold)):
            return True
    return False


def get_playing_area(image: np.ndarray, rect_size: tuple[int], templates: list[str] = None, pattern_threshold: float = 0.6,
                     color: tuple[int] = (0, 0, 255), display_steps: bool = False) -> tuple[np.ndarray, list[int]]:

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
        tiles_coords.append(((x, y), (x + w, y + h)))

    if display_steps:
        imshow(np.concatenate([gray, thresh], 1))
        imshow(cv2.resize(image_res, None, fx=0.8, fy=0.8))
    return image_res, tiles_coords


def create_masks(image:np.ndarray, coords_list:tuple[int]) -> list[np.ndarray]:
    mask_img_list = list()
    mask = np.zeros_like(image)
    for p1, p2 in coords_list:
        temp_mask = mask.copy()
        cv2.rectangle(temp_mask, (p1[0], p1[1]), (p2[0], p2[1]), (255, 255, 255), -1)

        # apply mask
        masked_image = cv2.bitwise_and(image, temp_mask)
        mask_img_list.append(masked_image)
    
    return mask_img_list


def take_masks_difference(image:np.ndarray, mask_img_list:list[np.ndarray]) -> np.ndarray:
    difference_img = np.zeros_like(image)
    for m in mask_img_list:
        subtracted = cv2.subtract(m, image)
        difference_img = cv2.bitwise_or(difference_img, subtracted)
    return difference_img