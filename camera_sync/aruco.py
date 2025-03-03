import cv2 as cv
import numpy as np
from camera import Camera
from position import refChange, invertRefChange, Point, Position

class Aruco:
    corners: list[Position] = None

    def __init__(self, id: int, size: float, topLeft: Position = None):
        self.id = id
        self.size = size
        if topLeft is not None:
            self.corners = self.setCornersFromTopLeft(topLeft)

    def setCornersFromTopLeft(self, topLeft: Position) -> list[Position]:
        self.corners = [
            topLeft,
            Position(topLeft.x + self.size, topLeft.y, topLeft.z),
            Position(topLeft.x + self.size, topLeft.y + self.size, topLeft.z),
            Position(topLeft.x, topLeft.y + self.size, topLeft.z),
        ]
        return self.corners

    def getCornersAsList(self) -> list[list[float]]:
        return np.array(
            [
                self.corners[0].coords,
                self.corners[1].coords,
                self.corners[2].coords,
                self.corners[3].coords,
            ]
        )

    def getCenter(self) -> Position:
        return Position(
            self.corners[0].x + self.size / 2,
            self.corners[0].y + self.size / 2,
            self.corners[0].z,
        )

    @property
    def isLocated(self) -> bool:
        return self.corners is not None


def generate_aruco_marker(
    marker_id,
    dictionary_id=cv.aruco.DICT_4X4_50,
    marker_size=200,
    save_path=None,
):
    """
    Generates an ArUco marker and saves it as an image.

    :param marker_id: The ID of the marker to generate (should be within the dictionary range)
    :param dictionary_id: The ArUco dictionary to use (default: DICT_4X4_50)
    :param marker_size: The size of the output image in pixels
    :param save_path: The file path to save the marker image
    :return: The generated marker image
    """
    if save_path is None:
        save_path = f"./aruco/aruco_{marker_id}.png"

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
        newPos = convertFromMarkerToWorld(aruco.corners[i].coords)
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
                final_image_points.append(c)

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

    if success:
        return rvec, tvec
    else:
        raise Exception("Could not solve PnP")
