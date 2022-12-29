import numpy as np
import cv2
from sklearn import cluster
import copy


params = cv2.SimpleBlobDetector_Params()
params.filterByInertia
params.minInertiaRatio = 0.6
detector = cv2.SimpleBlobDetector_create(params)


def get_blobs(frame: np.ndarray) -> list:
    frame_blurred = cv2.medianBlur(frame, 7)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    blobs = detector.detect(frame_gray)

    return blobs


def get_dice_from_blobs(blobs: list) -> list:
    X = []
    for b in blobs:
        pos = b.pt
        if pos != None:
            X.append(pos)

    X = np.asarray(X)

    if len(X) > 0:
        clustering = cluster.DBSCAN(eps=40, min_samples=1).fit(X)
        num_dice = max(clustering.labels_) + 1
        dice = []

        for i in range(num_dice):
            X_dice = X[clustering.labels_ == i]
            centroid_dice = np.mean(X_dice, axis=0)
            dice.append([len(X_dice), *centroid_dice])

        return dice

    else:
        return []


def overlay_info(frame: np.ndarray, dice: list, blobs: list, x: int) -> None:
    for b in blobs:
        pos = b.pt
        if pos[0] >= x:
            r = b.size / 2

            cv2.circle(frame, (int(pos[0]), int(pos[1])),
                    int(r), (255, 0, 0), 2)

    for d in dice:
        if d[1] >= x:
            textsize = cv2.getTextSize(
                str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

            cv2.putText(frame, str(d[0]),
                        (int(d[1] - textsize[0] / 2),
                        int(d[2] + textsize[1] / 2)),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)


def get_dice_number(frame: np.ndarray, x) -> np.ndarray:
    blobs = get_blobs(frame[:, x:])
    dice = get_dice_from_blobs(blobs)
    overlay_info(frame[:, x:], dice, blobs, x)
    return frame


def apply_mask(img: np.ndarray):
    img_masked = copy.deepcopy(img)
    hsv = cv2.cvtColor(img_masked, cv2.COLOR_BGR2HSV)

    sensitivity = 60
    lower_white = np.array([0, 0, 255-sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(img_masked, img_masked, mask=mask)

    return res


def get_contours(img: np.ndarray, area_min=10e4):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    contours = [c for n, c in enumerate(contours) if area_min < areas[n]]
    return contours


def get_rolling_area(img: np.ndarray):
    masked = apply_mask(img)
    masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    contours = get_contours(masked_gray)
    x, y, w, h = cv2.boundingRect(contours[0])

    return x
