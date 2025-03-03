from time import sleep
import numpy as np
import cv2 as cv
from aruco import calibrateCamera

import subprocess


def calibrateLiveCamera():
    print("Enter the camera ID:")
    cameraId = int(input())
    startCalibration(cameraId)


def findCameraId():
    """
    **NOT WORKING !**
    Cycle to all connected camera preview to locate the wanted camera system id
    """
    # !! Not working for now !!
    # TODO: To fix..
    print(
        "We will show you each camera feed until you find what you are looking for..."
    )
    done = False
    print("Press n to pass to next camera, press q to quit.")
    currCameraId = 0
    while not done:
        print("Showing video feed for camera ", currCameraId)
        cap = cv.VideoCapture(currCameraId)
        showingFeed = True
        while showingFeed:
            ret, frame = cap.read()
            if ret:
                cv.imshow("frame", frame)
                key = cv.waitKey(1)
                if key == ord("q"):
                    done = True
                    break
                elif key == ord("n"):
                    currCameraId += 1
                    break
            else:
                currCameraId += 1


def startCalibration(cameraId):
    """
    Calibrate a camera from a live camera feed

    :param
    cameraId - The id of the camera to calibrate
    """

    # Initialize the camera
    subprocess.run(
        ["v4l2-ctl", "-d", str(cameraId), "-c", "focus_automatic_continuous=0"]
    )
    sleep(0.5)
    subprocess.run(["v4l2-ctl", "-d", str(cameraId), "-c", "focus_absolute=0"])
    camera = cv.VideoCapture(cameraId)

    # camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    # camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

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
    calibrateLiveCamera()
