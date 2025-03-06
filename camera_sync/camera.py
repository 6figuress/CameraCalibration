import os
import subprocess
from time import sleep
import cv2 as cv
import numpy as np
from referential import Transform
from position import invertRefChange
from position import Position
from cv2.typing import MatLike, Vec3f


class Camera:
    deviceId: int
    captureStream: cv.VideoCapture
    mtx: MatLike
    dist: MatLike
    rvec: Vec3f
    tvec: Vec3f
    world_position: Position
    rotation_matrix: MatLike
    world2cam: Transform
    _focus: int
    _resolution: tuple[int, int]

    def __init__(
        self,
        name,
        deviceId,
        calibrationFolder: str = "./calibration",
        focus=0,
        resolution=(640, 480),
    ):
        if not os.path.exists(calibrationFolder):
            raise Exception(
                "Calibration folder not found. Searched at " + calibrationFolder
            )
        self.calibrationFolder = calibrationFolder

        self.name = name

        if deviceId != -1:
            self.deviceId = deviceId
            self.captureStream = cv.VideoCapture(deviceId)
            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    str(self.deviceId),
                    "-c",
                    "focus_automatic_continuous=0",
                ]
            )
            sleep(0.5)
            subprocess.run(
                ["v4l2-ctl", "-d", str(self.deviceId), "-c", f"focus_absolute={focus}"]
            )
            self.captureStream.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.captureStream.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.world_position = Position(0, 0, 0)
        self._focus = focus
        self._resolution = resolution

        if os.path.exists(self.calibrationFilePath):
            print(f"{self.name} : Found calibration file at {self.calibrationFilePath}")
            self._loadCalibration()
        else:
            print(
                f"{self.name} : Did not found any calibration file, searched at : {self.calibrationFilePath}"
            )

    @property
    def calibrationFilePath(self) -> str:
        return os.path.join(self.calibrationFolder, self.calibrationFileName)

    def _loadCalibration(self):
        file = np.load(self.calibrationFilePath)
        self.mtx = file["mtx"]
        self.dist = file["dist"]
        return file["mtx"], file["dist"]

    def calibrateWithLiveFeed(self, pointsToFind=(7, 7)):
        """
        Calibrate a camera from a live camera feed

        :param
        cameraId - The id of the camera to calibrate
        """

        done = False

        frames = []

        print(
            "Press 'c' to capture a frame, 's' to save the calibration to a file or 'q' to quit"
        )

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
                self.saveCalibration()
                break
            elif key == ord("c"):
                frames.append(frame)
                print("Frame added to calibration")
                self.calibrate(frames, pointsToFind)

    @property
    def calibrationFileName(self) -> str:
        return (
            f"{self.name}_{self._resolution[0]}-{self._resolution[1]}_{self._focus}.npz"
        )

    def saveCalibration(self):
        """
        Save the current camera calibration to a file for futur use

        Args:
            filePath: The path to the file to save the calibration to
        """

        np.savez(
            self.calibrationFilePath,
            mtx=self.mtx,
            dist=self.dist,
        )
        print(f"Calibration file saved in {self.calibrationFilePath}")

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
        objp[:, :2] = np.mgrid[0 : pointsToFind[0], 0 : pointsToFind[1]].T.reshape(
            -1, 2
        )

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

    def updateWorldPosition(self, rvec: Vec3f, tvec: Vec3f) -> tuple[Vec3f, MatLike]:
        self.rvec = rvec
        self.tvec = tvec.flatten()
        self.rotation_matrix, _ = cv.Rodrigues(rvec)

        self.world2cam = Transform(rvec=rvec, tvec=tvec)

        self.world_position.updatePos(
            invertRefChange(np.array([0.0, 0.0, 0.0]), self.rotation_matrix, self.tvec)
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
    camera = Camera("Logitec_A", 2, focus=0, resolution=(1280, 720))
    camera.calibrateWithLiveFeed()
    cv.destroyAllWindows()
