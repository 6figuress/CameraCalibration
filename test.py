from matplotlib import animation
from aruco import *
import cv2 as cv


def takePicture(camID):
    cap = cv.VideoCapture(camID)
    ret, frame = cap.read()
    cap.release()
    return frame


def test():
    img = cv.imread("loca/loca_logi.jpg")

    cam = Camera(0, "calibration/logitec_2_f30.npz")

    arucos: dict[int, Aruco] = {
        10: Aruco(10, Position(10, 15, 0), size=80),
        11: Aruco(11, Position(120, 15, 0), size=80),
        12: Aruco(12, Position(10, 110, 0), size=80),
        13: Aruco(13, Position(120, 110, 0), size=80),
        14: Aruco(14, Position(10, 205, 0), size=80),
        15: Aruco(15, Position(120, 205, 0), size=80),
    }

    fixed = [10, 11, 14, 15]

    fixedArucos = {}

    for i in fixed:
        fixedArucos[i] = arucos[i]

    locate = [12, 13]
    locateArucos = {}

    for i in locate:
        locateArucos[i] = arucos[i]

    rvec, tvec, locatedArucos = processAruco(
        fixedArucos.values(), locateArucos.values(), cam, img
    )

    position, R = locateCameraWorld(rvec, tvec)

    import ipdb

    ipdb.set_trace()


test()

# img = takePicture(0)

# cv.imwrite("loca/loca_logi.jpg", img)
