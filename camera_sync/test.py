from camera import Camera
from aruco import processAruco, Aruco
from position import Position
import cv2 as cv

arucos: dict[int, Aruco] = {
    10: Aruco(10, size=80, topLeft=Position(10, 15, 0)),
    11: Aruco(11, size=80, topLeft=Position(120, 15, 0)),
    12: Aruco(12, size=80, topLeft=Position(10, 110, 0)),
    13: Aruco(13, size=80, topLeft=Position(120, 110, 0)),
    14: Aruco(14, size=80, topLeft=Position(10, 205, 0)),
    15: Aruco(15, size=80, topLeft=Position(120, 205, 0)),
}

cam = Camera(2, "./calibration/logitec_2_f30.npz")

baseFrame = cam.takePic()

anchors = [
    arucos[10],
    arucos[11],
    arucos[14],
    arucos[15],
    arucos[13]
]

rvec, tvec, located = processAruco(anchors, [], cam, baseFrame)



cv.imwrite("test.png", cam.undistort(baseFrame))


print("Position : ")
print("X : ", cam.world_position[0])
print("Y : ", cam.world_position[1])
print("Z : ", cam.world_position[2])



print("Rotation : ")
print("X : ", cam.rvec[0])
print("Y : ", cam.rvec[1])
print("Z : ", cam.rvec[2])

# Position :
# X: -268.788
# Y: -141.022
# Z: -380.272

# Rotation:
# X: -0.61747755
# Y: -1.02261949
# Z: -1.97218344
