import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


class Camera:
    deviceId: int
    captureStream: cv.VideoCapture
    mtx: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    world_position: np.ndarray
    rotation_matrix: np.ndarray

    def __init__(self, deviceId, calibrationFile):
        self.deviceId = deviceId
        self.calibrationFile = calibrationFile
        self.captureStream = cv.VideoCapture(deviceId)
        # self.captureStream.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        # self.captureStream.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        self.mtx, self.dist = loadCalibration(calibrationFile)

    def updateWorldPosition(self, rvec, tvec):
        self.rvec = rvec
        self.tvec = tvec.flatten()
        self.rotation_matrix, _ = cv.Rodrigues(rvec)
        self.world_position = invertRefChange(
            np.array([0.0, 0.0, 0.0]), self.rotation_matrix, self.tvec
        )
        return self.world_position, self.rotation_matrix


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Position(Point):
    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y)
        self.z = z

    def toList(self):
        return [self.x, self.y, self.z]

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}({self.x}, {self.y}, {self.z})"

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"


class Aruco:
    corners: list[Position]

    def __init__(self, id: int, topLeft: Position, size: float = 1.0):
        self.id = id
        self.size = size
        self.corners = self.setCornersFromTopLeft(topLeft)

    def setCornersFromTopLeft(self, topLeft: Position):
        corners = [
            topLeft,
            Position(topLeft.x + self.size, topLeft.y, topLeft.z),
            Position(topLeft.x + self.size, topLeft.y + self.size, topLeft.z),
            Position(topLeft.x, topLeft.y + self.size, topLeft.z),
        ]
        return corners

    def getCornersAsList(self):
        return np.array(
            [
                self.corners[0].toList(),
                self.corners[1].toList(),
                self.corners[2].toList(),
                self.corners[3].toList(),
            ]
        )

    def getCenter(self):
        return Position(
            self.corners[0].x + self.size / 2,
            self.corners[0].y + self.size / 2,
            self.corners[0].z,
        )


def refChange(position: np.ndarray, rot_mat, tvec):
    return rot_mat @ position + tvec


def invertRefChange(position: np.ndarray, rot_mat, tvec):
    inv_tvec = -rot_mat.T @ tvec
    return rot_mat.T @ position + inv_tvec


def generate_aruco_marker(
    marker_id,
    dictionary_id=cv.aruco.DICT_4X4_50,
    marker_size=200,
    save_path=f"./aruco/aruco_marker.png",
):
    """
    Generates an ArUco marker and saves it as an image.

    :param marker_id: The ID of the marker to generate (should be within the dictionary range)
    :param dictionary_id: The ArUco dictionary to use (default: DICT_4X4_50)
    :param marker_size: The size of the output image in pixels
    :param save_path: The file path to save the marker image
    :return: The generated marker image
    """
    # Load the predefined dictionary
    aruco_dict = cv.aruco.getPredefinedDictionary(dictionary_id)

    # Create an empty image to store the marker
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)

    # Generate the marker
    marker_image = cv.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # Save the marker image
    cv.imwrite(save_path, marker_image)
    print(f"ArUco marker (ID: {marker_id}) saved as {save_path}")
    return marker_image


def calibrateCamera(images, pointsToFind=(7, 7), filepath=None):
    """
    Calibrate the camera using a set of images of a chessboard pattern.

    :param images: A list of image file paths
    :param pointsToFind: The number of points to find on the chessboard pattern
    :param filepath: The file path to save the calibration matrix
    :return: The camera matrix and distortion coefficients
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
    for i, img in enumerate(images):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, pointsToFind, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            print(f"Found {len(corners2)} corners - pic {i + 1}/{len(images)}")
            imgpoints.append(corners)

            # Draw and display the corners
            # img = cv.drawChessboardCorners(img, pointsToFind, corners2, ret)

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
    if filepath is not None:
        np.savez(filepath, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print(f"Calibration done. Calibration matrix saved in {filepath}")

    return mtx, dist


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


def loadCalibration(path):
    file = np.load(path)
    return file["mtx"], file["dist"]


def drawArucoCorners(frame, img_points: list[Point]):
    colors = [
        (0, (255, 0, 0)),
        (1, (0, 255, 0)),
        (2, (0, 0, 255)),
        (3, (255, 255, 0)),
    ]

    for a in img_points:
        for c in colors:
            cv.circle(
                frame,
                (int(a[c[0]][0]), int(a[c[0]][1])),
                10,
                c[1],
                -1,
            )


def detectAruco(
    img, dictionary_id=cv.aruco.DICT_4X4_50, debug=False
) -> tuple[dict, list]:
    """
    Detect ArUco markers in an image.

    :param img: The image to detect markers in
    :param dictionary_id: The ArUco dictionnary to detect
    :param debug: Whether to display debug information
    :return: A dictionary of detected markers and their corners, and a list of rejected markers
    """

    detected = {}

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    arucoDict = cv.aruco.getPredefinedDictionary(dictionary_id)

    arucoParams = cv.aruco.DetectorParameters()
    (corners, ids, rejected) = cv.aruco.detectMarkers(
        gray, arucoDict, parameters=arucoParams
    )

    if ids is None:
        if debug:
            print("No markers found")
        return detected, rejected

    for i, id in enumerate(ids):
        detected[id[0]] = corners[i]
    if debug:
        print("Detected aruco markers : ", detected.keys())
    return detected, rejected


def locateAruco(aruco: Aruco, img_positions: list, camera: Camera):

    assert len(img_positions[0]) == 4

    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
        img_positions, aruco.size, camera.mtx, camera.dist
    )

    rvec = rvecs[0][0]  # Extract the rotation vector
    tvec = tvecs[0][0]  # Extract the translation vector

    topLeft = Position(-aruco.size / 2, -aruco.size / 2, 0)

    aruco.corners = aruco.setCornersFromTopLeft(topLeft)

    R, _ = cv.Rodrigues(rvec)

    def convertFromMarkerToWorld(pos):
        pos = refChange(pos, R, tvec)
        pos = invertRefChange(pos, camera.rotation_matrix, camera.tvec)
        return pos

    for i in range(len(aruco.corners)):
        newPos = convertFromMarkerToWorld(aruco.corners[i].toList())
        aruco.corners[i] = Position(*newPos)

    return aruco


def processAruco(
    fixedArucos: list[Aruco],
    movingArucos: list[Aruco],
    camera: Camera,
    img,
    accept_none=False,
):
    corners_position, rejected = detectAruco(img, debug=False)
    final_image_points = []
    final_obj_points = []
    for aruco in fixedArucos:
        if corners_position.get(aruco.id) is None:
            continue
        else:
            for c in aruco.getCornersAsList():
                final_obj_points.append(c)
            for c in corners_position[aruco.id][0]:
                final_image_points.append(c.tolist())

    if len(final_image_points) < 4 or len(final_obj_points) < 4:
        if accept_none:
            return False
        raise Exception("Could not find any ArUco markers in the image")

    final_image_points = np.array(final_image_points)
    final_obj_points = np.array(final_obj_points)

    rvec, tvec = PnP(
        np.array(final_image_points, dtype=np.float32),
        np.array(final_obj_points, dtype=np.float32),
        camera.mtx,
        camera.dist,
    )

    camera.updateWorldPosition(rvec, tvec)

    locatedArucos: dict[int, Aruco] = {}
    for a in movingArucos:
        if corners_position.get(a.id) is None:
            continue
        else:
            locatedArucos[a.id] = locateAruco(a, corners_position[a.id], camera)

    return rvec, tvec, locatedArucos


def PnP(image_points, object_points, mtx, dist) -> tuple[list, list]:
    """
    Estimate the pose of an the camera using PnP.

    :param points: The 2D points of the object in the image
    :param object_points: The 3D points of the object
    :param mtx: The camera matrix
    :param dist: The distortion coefficients
    :return: The rotation and translation vectors
    """

    image_points = np.array(image_points, dtype=np.float32)

    # Solve PnP to estimate rotation and translation

    try:
        success, rvec, tvec = cv.solvePnP(object_points, image_points, mtx, dist)
    except cv.error as e:
        print("An error occured while solving PnP")
        print(
            f"Object point n°: {len(object_points)}, Image point n°: {len(image_points)}"
        )
        raise e
    # success, rvec, tvec, inliers = cv.solvePnPRansac(
    #     object_points, image_points, mtx, dist
    # )
    if success:
        # Project a new 3D point using the estimated pose
        # new_3d_point = np.array(
        #     [(0.5, 0.5, 0.0)], dtype=np.float32
        # )  # Center of the square
        # projected_2d, _ = cv.projectPoints(new_3d_point, rvec, tvec, mtx, dist)
        return rvec, tvec
    else:
        print("Could not solve PnP")
        raise Exception("Could not solve PnP")
