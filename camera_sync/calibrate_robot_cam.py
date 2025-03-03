import os
from camera import Camera
import cv2 as cv
import numpy as np


def calibrate(pics, pointsToFind=(7, 7)):
    """
    Calibrate the camera using a set of images of a chessboard pattern.

    Parameters:
        pics: A list of image file paths
        pointsToFind: The number of points to find on the chessboard pattern
        filePath: The file path to save the calibration matrix
    Returns:
        The camera matrix and distortion coefficients
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pointsToFind[0] * pointsToFind[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pointsToFind[0], 0 : pointsToFind[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    print("Starting calibration, this may take some time...")
    for i, img in enumerate(pics):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, pointsToFind, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    def evaluateCalibration():
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        print("total error: {}".format(mean_error / len(objpoints)))

    evaluateCalibration()

    return mtx, dist


pics = []

files = os.listdir("./pics")


for f in files:
    pics.append(cv.imread("./pics/" + f))


mtx, dist = calibrate(pics)

np.savez("./calibration/robot_cam.npz", mtx=mtx, dist=dist)

import ipdb

ipdb.set_trace()
