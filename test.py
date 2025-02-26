from matplotlib import animation
from aruco import *
import cv2 as cv

img = cv.imread("loca/loca_logi.jpg")

mtx, dist = loadCalibration("calibration/logitec_2_f30.npz")

FixedArucos: dict[int, Aruco] = {
    0: Aruco(0, Position(0, 0, 0), size=3),
    1: Aruco(1, Position(15, 3, 0), size=3),
    2: Aruco(2, Position(3, 18, 0), size=3),
}


realPosition = {
    3: Aruco(3, Position(18, 19, 0), size=3),
}

movingArucos = [3]


rvec, tvec, locatedArucos = processAruco(
    FixedArucos.values(), movingArucos, mtx, dist, img
)


for c in locatedArucos[3].corners:
    print("Aruco corner location : ", c.x, c.y, c.z)

import ipdb

ipdb.set_trace()
