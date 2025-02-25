import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Position(Point):
    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y)
        self.z = z


class Aruco:
    corners: list[Position]

    def __init__(self, id: int, topLeft: Position, size: float = 1.0):
        self.id = id
        self.corners = [
            topLeft,
            Position(topLeft.x + size, topLeft.y, topLeft.z),
            Position(topLeft.x + size, topLeft.y + size, topLeft.z),
            Position(topLeft.x, topLeft.y + size, topLeft.z),
        ]
        self.size = size


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


def calibrateCamera(
    images, pointsToFind=(7, 7), filepath="./calibration/calibration.npz"
):
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
    for i, fname in enumerate(images):
        img = cv.imread(fname)
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
            img = cv.drawChessboardCorners(img, pointsToFind, corners2, ret)
    cv.destroyAllWindows()

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

    np.savez(filepath, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    print(f"Calibration done. Calibration matrix saved in {filepath}")

    return mtx, dist


def undistort(mtx, dist, img):
    """
    Undistort an image using the camera matrix and distortion coefficients

    :param mtx: The camera matrix
    :param dist: The distortion coefficients
    :param img: The image to undistort
    :return: The undistorted image
    """

    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    return dst


def loadCalibration(path):
    file = np.load(path)
    return file["mtx"], file["dist"]


def drawArucoCorners(frame, arucos: list[Point]):
    colors = [
        (0, (255, 0, 0)),
        (1, (0, 255, 0)),
        (2, (0, 0, 255)),
        (3, (255, 255, 0)),
    ]

    for a in arucos:
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
        print("No markers found")
        return detected, rejected

    for i, id in enumerate(ids):
        detected[id[0]] = corners[i]
    if debug:
        colors = [
            (0, (255, 0, 0)),
            (1, (0, 255, 0)),
            (2, (0, 0, 255)),
            (3, (255, 255, 0)),
        ]

        for c in corners:
            for l in colors:
                cv.circle(img, (int(c[0][l[0]][0]), int(c[0][l[0]][1])), 10, l[1], -1)
                print("Color is : ", l[1])
        print("Detected aruco markers : ", detected.keys())
        cv.imwrite("debug_aruco.png", img)
    return detected, rejected


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

    success, rvec, tvec = cv.solvePnP(object_points, image_points, mtx, dist)
    # success, rvec, tvec, inliers = cv.solvePnPRansac(
    #     object_points, image_points, mtx, dist
    # )
    if success:
        # Project a new 3D point using the estimated pose
        new_3d_point = np.array(
            [(0.5, 0.5, 0.0)], dtype=np.float32
        )  # Center of the square
        projected_2d, _ = cv.projectPoints(new_3d_point, rvec, tvec, mtx, dist)
        return rvec, tvec
    else:
        print("Could not solve PnP")
        raise Exception("Could not solve PnP")


def draw_camera_position(ax, position, rotation):
    point = ax.scatter(*position.flatten(), color="black")
    camera_direction = rotation[:, 2]
    endpoint = position.flatten() + 10 * camera_direction
    # Plot the camera direction of looking
    line = ax.plot(
        [position[0][0], endpoint[0]],
        [position[1][0], endpoint[1]],
        [position[2][0], endpoint[2]],
        color="purple",
    )
    return point, line


def init_3d_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xbound(-50, 50)
    ax.set_ybound(-50, 50)
    ax.set_zbound(-50, 50)
    return ax, fig


def draw_arucos(ax, arucos):
    for aruco in arucos:
        # Draw the aruco id on top of the marker
        ax.text(aruco.corners[0].x, aruco.corners[0].y - 1, 0.0, aruco.id, fontsize=20)
        for i in range(0, len(aruco.corners)):
            # Draw a square corresponding to the marker position
            ax.plot(
                [aruco.corners[i].x, aruco.corners[(i + 1) % 4].x],
                [aruco.corners[i].y, aruco.corners[(i + 1) % 4].y],
                [aruco.corners[i].z, aruco.corners[(i + 1) % 4].z],
                color="black",
            )


def plot_camera_pose(positions: list, rotations: list, arucos: list[Aruco]):
    """
    Plot the camera pose and object points in 3D.

    :param positions: The camera positions
    :param rotations: The camera rotations
    :param arucos: The arucos
    """

    ax, fig = init_3d_plot()

    # Plot camera positions
    for i, p in enumerate(positions):
        ax.scatter(*p.flatten(), color="black", label=f"Camera Position {i}")
        camera_direction = rotations[i][:, 2]
        endpoint = p.flatten() + 10 * camera_direction
        # Plot the camera direction of looking
        ax.plot(
            [p[0][0], endpoint[0]],
            [p[1][0], endpoint[1]],
            [p[2][0], endpoint[2]],
            color="purple",
        )

    draw_arucos(ax, arucos)

    ax.legend()
    plt.show()


def getPosition(img, mtx, dist, aruco_positions: dict[int, Aruco]):
    """
    Get the position of the camera from an image containing an ArUco marker.

    :param img: The image
    :param mtx: The camera matrix
    :param dist: The distortion coefficients
    :param object_points: The 3D object points
    :return: The rotation and translation
    """
    arucos, rejected = detectAruco(img)
    final_image_points = []
    final_obj_points = []

    for id in aruco_positions.keys():
        if arucos.get(id) is None:
            print(f"Marker {id} not found in picture !")
        else:
            aruco = aruco_positions[id]
            for c in aruco.corners:
                final_obj_points.append([c.x, c.y, c.z])
            for c in arucos[id][0]:
                final_image_points.append(c.tolist())

    final_image_points = np.array(final_image_points)
    final_obj_points = np.array(final_obj_points)

    assert len(final_image_points) > 0 and len(final_obj_points) > 0

    assert len(final_image_points) == len(final_obj_points)
    print("Found ", len(final_image_points), " points")
    rvec, tvec = PnP(
        np.array(final_image_points), np.array(final_obj_points), mtx, dist
    )
    return rvec, tvec


if __name__ == "__main__":
    # Example of camera localization in 3d using 4 aruco markers
    # This is the linking between aruco markers for the result graph

    # This is the positions of the aruco markers in the 3d world
    aruco_positions: dict[int, Aruco] = {
        # 0: aruco(0, Position(0, 0, 0)),
        # 1: aruco(1, Position(15, 3, 0)),
        # 2: aruco(2, Position(3, 18, 0)),
        3: Aruco(3, Position(18, 19, 0)),
    }

    mtx, dist = loadCalibration("./calibration/calibration.npz")

    rvec, tvec = getPosition("./loca/0.jpg", mtx, dist, aruco_positions)
    R, _ = cv.Rodrigues(rvec)
    camera_position = -R.T @ tvec
    plot_camera_pose([camera_position], [R], aruco_positions.values())
