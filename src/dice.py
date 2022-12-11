import numpy as np
import cv2
from sklearn import cluster


params = cv2.SimpleBlobDetector_Params()
params.filterByInertia
params.minInertiaRatio = 0.6
detector = cv2.SimpleBlobDetector_create(params)


def get_blobs(frame):
    frame_blurred = cv2.medianBlur(frame, 7)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    blobs = detector.detect(frame_gray)

    return blobs


def get_dice_from_blobs(blobs):
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


def overlay_info(frame, dice, blobs):
    for b in blobs:
        pos = b.pt
        r = b.size / 2

        cv2.circle(frame, (int(pos[0]), int(pos[1])),
                   int(r), (255, 0, 0), 2)

    for d in dice:
        textsize = cv2.getTextSize(
            str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(frame, str(d[0]),
                    (int(d[1] - textsize[0] / 2),
                     int(d[2] + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        # print(d[0])
        # d[0] - number on die
