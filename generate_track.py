import math
import os
import cv2
import numpy as np


def get_perpendicular_coordinates(p1, p2, length):
    vX = p2[0] - p1[0]
    vY = p2[1] - p1[1]
    if (vX == 0 or vY == 0):
        return 0, 0, 0, 0
    mag = math.sqrt(vX * vX + vY * vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0 - vY
    vY = temp
    cX = p2[0] + vX * length
    cY = p2[1] + vY * length
    dX = p2[0] - vX * length
    dY = p2[1] - vY * length
    return (int(cX), int(cY)), (int(dX), int(dY))


def calculate_boundaries(centerline, track_width=10):
    inner = []
    outer = []
    cl = np.asarray(centerline)
    for i, point in enumerate(cl, start=1):
        if i < len(cl):
            p1 = np.asarray(centerline[i - 1][0])
            p2 = np.asarray(centerline[i][0])
            inner_point, outer_point = get_perpendicular_coordinates(p1, p2, track_width)
            inner.append(inner_point)
            outer.append(outer_point)

    return np.array(inner, dtype=int, copy=False), np.array(outer, dtype=int, copy=False)


def detect_contour(img, desired_points=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 1:
        point_count = len(cnts[0])
        print('initial contour length: {}'.format(point_count))
        if desired_points is None or desired_points > point_count:
            peri = cv2.arcLength(cnts[0], True)
            approx = cv2.approxPolyDP(cnts[0], 0.0005 * peri, True)
            return approx
        else:

            approx = []
            mod = int(round(point_count / desired_points))
            for i, point in enumerate(cnts[0], start=0):
                if i % mod == 0:
                    approx.append(point)

            return np.array(approx, dtype=int, copy=False)


image = cv2.imread(os.path.join(os.curdir, 'shapes', 'track-shape.png'))

center_line = detect_contour(image, desired_points=200)
print('smoothed contour length: {}'.format(len(center_line)))
inner, outer = calculate_boundaries(center_line, 30)

shape = image.shape

blank_image = np.zeros(shape, np.uint8)

cv2.drawContours(blank_image, [inner], -1, (255, 0, 0), 4)
cv2.drawContours(blank_image, [outer], -1, (255, 0, 0), 4)
cv2.drawContours(blank_image, [center_line], -1, (12, 215, 255), 4)
cv2.imshow("Track", blank_image)
cv2.waitKey(0)
