import os
import cv2 as cv
import numpy as np
from camera import Camera
from referential import Transform
from scipy.spatial.transform import Rotation as R
from aruco import processAruco, getArucosFromPaper

def create_blender_camera_matrix(image_width, image_height, focal_length_mm=35, sensor_width_mm=36):
    """
    Create a camera matrix for a Blender camera based on its parameters.
    
    Parameters:
        image_width: Image width in pixels
        image_height: Image height in pixels
        focal_length_mm: Focal length in mm (default Blender camera is 35mm)
        sensor_width_mm: Sensor width in mm (default Blender camera is 36mm)
    
    Returns:
        Camera matrix as a 3x3 numpy array
    """
    # Calculate pixel size
    pixel_size_mm = sensor_width_mm / image_width
    
    # Calculate focal length in pixels
    focal_length_px = focal_length_mm / pixel_size_mm
    
    # Principal point (usually at the center of the image)
    cx = image_width / 2
    cy = image_height / 2
    
    # Create camera matrix
    camera_matrix = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return camera_matrix

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


# pics = []

# baseFolder = "./pics/blender/calib"

# for f in os.listdir(baseFolder):
#     pics.append(cv.imread(os.path.join(baseFolder, f)))

# mtx, dist = calibrate(pics)

# np.savez("./calibration/blender_1920-1080_0.npz", mtx=mtx, dist=dist)

# Define your image dimensions
width, height = 1920, 1080

# Create camera matrix directly (using Blender's default parameters)
camera_matrix = create_blender_camera_matrix(width, height)

# Blender's rendered images have no distortion
dist_coeffs = np.zeros((1, 5), dtype=np.float32)

# Create camera with the calculated matrix
blend_cam = Camera("blender", -1, focus=0, resolution=(width, height))
blend_cam.mtx = camera_matrix
blend_cam.dist = dist_coeffs

arucos = getArucosFromPaper()

poses = [
    [
        [0, 0, 800],
        [200, -200, 800],
        [
            661.144,
            -130.141,
            373.113,
        ],
    ],
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        R.from_quat([0.322, 0.310, 0.620, 0.644]).as_rotvec(),
    ],
]

frame = cv.imread("./pics/blender/poses/pose2.png")

rvec, tvec, _, _ = processAruco(arucos.values(), [], blend_cam, frame)


import ipdb

ipdb.set_trace()
