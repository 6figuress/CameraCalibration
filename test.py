from matplotlib import animation
from aruco import *
import cv2 as cv

# All four markers presents
aruco_positions: dict[int, Aruco] = {
    0: Aruco(0, Position(0, 0, 0)),
    1: Aruco(1, Position(15, 3, 0)),
    2: Aruco(2, Position(3, 18, 0)),
    3: Aruco(3, Position(18, 19, 0)),
}

mtx, dist = loadCalibration("./calibration/calibration.npz")


cap = cv.VideoCapture("./loca/video.mp4")

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

write = cv.VideoWriter(
    "output.avi", cv.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height)
)
positions = []
rotations = []
while True:

    ret, frame = cap.read()
    if not ret:
        break
    arucos, rejected = detectAruco(frame)
    arucos = list(arucos.values())
    for i in range(0, len(arucos)):
        arucos[i] = arucos[i][0]

    drawArucoCorners(frame, arucos)
    write.write(frame)

cap.release()
write.release()
