from time import sleep
import cv2 as cv
from aruco import *

import subprocess


def startCalibration(cameraId):
    # Initialize the camera
    subprocess.run(
        ["v4l2-ctl", "-d", str(cameraId), "-c", "focus_automatic_continuous=0"]
    )
    sleep(0.5)
    subprocess.run(["v4l2-ctl", "-d", str(cameraId), "-c", "focus_absolute=0"])
    camera = cv.VideoCapture(cameraId)

    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    done = False

    frames = []

    print("Press 'c' to capture a frame, 'q' to quit")

    while not done:

        ret, frame = camera.read()

        if not ret:
            print("Failed to capture frame")
            continue

        cv.imshow("frame", frame)
        print("Frame : ", frame.shape)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("c"):
            frames.append(frame)
            print("Frame added to calibration")
            mtx, dist = calibrateCamera(frames, (7, 7))

    # Release the camera
    camera.release()

    # Save the calibration
    np.savez("./calibration/calibration.npz", mtx=mtx, dist=dist)


if __name__ == "__main__":
    print("Enter the camera ID:")
    cameraId = int(input())
    startCalibration(cameraId)
