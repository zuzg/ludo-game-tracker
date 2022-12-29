import numpy as np
import cv2
from src.utils import *


def match_patterns(board: np.ndarray, templates: list[str], color: tuple[int]) -> tuple[np.ndarray, list[int]]:
    coords = list()
    board_matched = board.copy()
    board_gray = cv2.cvtColor(board_matched, cv2.COLOR_BGR2GRAY)
    for t in templates:
        template = cv2.imread(t, 1)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = template.shape[0], template.shape[1]
        corr = cv2.matchTemplate(board_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)

        cv2.rectangle(board_matched, (max_loc[0], max_loc[1]), (max_loc[0]+template.shape[1], max_loc[1]+template.shape[0]), color, 2)
        coords.append(((max_loc[0], max_loc[1]), (max_loc[0]+template.shape[1], max_loc[1]+template.shape[0])))

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
