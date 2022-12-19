import numpy as np
import cv2


def match_patterns(board, templates, color):
    board_matched = board.copy()
    for t in templates:
        template = cv2.imread(f"./data/templates/{t}.jpg", 1)
        board_gray = cv2.cvtColor(board_matched, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        w, h = template.shape[0], template.shape[1]
        corr = cv2.matchTemplate(board_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        threshhold = 0.8
        loc = np.where(corr >= threshhold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(board_matched, pt, (pt[0] + w, pt[1] + h), color, 2)

    return board_matched
