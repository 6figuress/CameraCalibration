import subprocess
from time import sleep
import cv2 as cv
import numpy as np
from position import invertRefChange

class Camera:
    deviceId: int
    captureStream: cv.VideoCapture
    mtx: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    world_position: np.ndarray
    rotation_matrix: np.ndarray

    def __init__(self, deviceId, calibrationFile: str = None):
        self.deviceId = deviceId
        self.calibrationFile = calibrationFile
        self.captureStream = cv.VideoCapture(deviceId)
        # self.captureStream.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        # self.captureStream.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        if calibrationFile is not None:
            self._loadCalibration(calibrationFile)

    def _loadCalibration(self, path):
        file = np.load(path)
        self.mtx = file["mtx"]
        self.dist = file["dist"]
        return file["mtx"], file["dist"]



    def calibrateWithLiveFeed(self, pointsToFind=(7,7), filePath = None):
        """
        Calibrate a camera from a live camera feed

        :param
        cameraId - The id of the camera to calibrate
        """

        if filePath is None:
            filePath = "./calibration/new_calibration.npz"

        # Initialize the camera
        subprocess.run(
            ["v4l2-ctl", "-d", str(self.deviceId), "-c", "focus_automatic_continuous=0"]
        )
        sleep(0.5)
        subprocess.run(["v4l2-ctl", "-d", str(self.deviceId), "-c", "focus_absolute=30"])

        # camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        # camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

        done = False

        frames = []

        print("Press 'c' to capture a frame, 's' to save the calibration to a file or 'q' to quit")

        while not done:

            ret, frame = self.captureStream.read()

            if not ret:
                print("Failed to capture frame")
                continue

            cv.imshow("frame", frame)
            key = cv.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("s"):
                self.saveCalibration(filePath)
                break
            elif key == ord("c"):
                frames.append(frame)
                print("Frame added to calibration")
                self.calibrate(frames, (7, 7))

    
    def saveCalibration(self, filePath:str):
        """
        Save the current camera calibration to a file for futur use

        Args:
            filePath: The path to the file to save the calibration to 
        """

        np.savez(filePath, mtx=self.mtx, dist=self.dist)
        print(f"Calibration done. Calibration matrix saved in {filePath}")

    def calibrate(self, pics, pointsToFind=(7, 7)):
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

        self.mtx = mtx
        self.dist = dist

        return mtx, dist

    def updateWorldPosition(self, rvec, tvec):
        self.rvec = rvec
        self.tvec = tvec.flatten()
        self.rotation_matrix, _ = cv.Rodrigues(rvec)
        self.world_position = invertRefChange(
            np.array([0.0, 0.0, 0.0]), self.rotation_matrix, self.tvec
        )
        return self.world_position, self.rotation_matrix


def undistort(camera: Camera, img):
    """
    Undistort an image using the camera matrix and distortion coefficients

    :param mtx: The camera matrix
    :param dist: The distortion coefficients
    :param img: The image to undistort
    :return: The undistorted image
    """

    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        camera.mtx, camera.dist, (w, h), 1, (w, h)
    )

    # undistort
    dst = cv.undistort(img, camera.mtx, camera.dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    return dst

if __name__ == "__main__":
    camera = Camera(2)
    camera.calibrateWithLiveFeed()
    camera.saveCalibration("./calib.npz")
    cv.destroyAllWindows()