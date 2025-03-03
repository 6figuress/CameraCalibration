import cv2 as cv
import numpy as np
from camera import Camera
from position import refChange, invertRefChange, Point, Position
from itertools import combinations


class Aruco:
    corners: list[Position] = None

    @staticmethod
    def getCornersFromTopLeft(
        topLeft: np.ndarray[float], size
    ) -> np.ndarray[np.ndarray[float]]:
        topLeft
        topRight = topLeft.copy()
        topRight[0] += size
        bottomRight = topLeft.copy()
        bottomRight[0] += size
        bottomRight[1] += size
        bottomLeft = topLeft.copy()
        bottomLeft[1] += size
        return np.array([topLeft, topRight, bottomRight, bottomLeft])

    def __init__(self, id: int, size: float, topLeft: Position = None):
        self.id = id
        self.size = size
        if topLeft is not None:
            self.corners = [
                Position(*c)
                for c in Aruco.getCornersFromTopLeft(topLeft.coords, self.size)
            ]

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


def getArucosFromPaper() -> dict[int, Aruco]:
    return {
        1: Aruco(1, size=27, topLeft=Position(0, 0, 0)),
        2: Aruco(2, size=27, topLeft=Position(0, 0, 0)),
        3: Aruco(3, size=27, topLeft=Position(0, 0, 0)),
        0: Aruco(0, size=27, topLeft=Position(0, 0, 0)),
        10: Aruco(10, size=80, topLeft=Position(10, 15, 0)),
        11: Aruco(11, size=80, topLeft=Position(120, 15, 0)),
        12: Aruco(12, size=80, topLeft=Position(10, 110, 0)),
        13: Aruco(13, size=80, topLeft=Position(120, 110, 0)),
        14: Aruco(14, size=80, topLeft=Position(10, 205, 0)),
        15: Aruco(15, size=80, topLeft=Position(120, 205, 0)),
    }


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
    # TODO: We may need to test different methods here to find the most accurate for our case !
    arucoParams.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    arucoParams.cornerRefinementWinSize = 20
    arucoParams.cornerRefinementMinAccuracy = 0.01
    arucoParams.cornerRefinementMaxIterations = 100
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


def locateAruco(
    aruco: Aruco, img_positions: list, camera: Camera, metrics: bool = False
):
    metrics = {}

    assert len(img_positions[0]) == 4

    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
        img_positions, aruco.size, camera.mtx, camera.dist
    )

    rvec = rvecs[0][0]  # Extract the rotation vector
    tvec = tvecs[0][0]  # Extract the translation vector

    corners = Aruco.getCornersFromTopLeft(
        np.array([-aruco.size / 2, -aruco.size / 2, 0]), aruco.size
    )

    R, _ = cv.Rodrigues(rvec)

    def convertFromMarkerToWorld(pos):
        pos = refChange(pos, R, tvec)
        pos = invertRefChange(pos, camera.rotation_matrix, camera.tvec)
        return pos

    world_corners = []

    for i, c in enumerate(corners):
        world_corners.append(convertFromMarkerToWorld(c))

    return world_corners, metrics


def processAruco(
    fixedArucos: list[Aruco],
    movingArucos: list[Aruco],
    camera: Camera,
    img,
    accept_none=False,
    directUpdate=True,
    metrics=False,
):
    metrics_collected = {}
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

    rvec, tvec, met = PnP(
        np.array(final_image_points, dtype=np.float32),
        np.array(final_obj_points, dtype=np.float32),
        camera.mtx,
        camera.dist,
        metrics,
    )

    if metrics:
        metrics_collected["PnP"] = met

    camera.updateWorldPosition(rvec, tvec)

    arucosPosition: dict[int, np.ndarray[np.ndarray[float]]] = {}

    for a in movingArucos:
        if corners_position.get(a.id) is None:
            continue
        else:
            newCorners, ar_met = locateAruco(a, corners_position[a.id], camera)
            if directUpdate:
                for i, c in enumerate(newCorners):
                    a.corners[i].updatePos(c)
            arucosPosition[a.id] = newCorners

    return rvec, tvec, arucosPosition, metrics_collected


def PnP(image_points, object_points, mtx, dist, getMetrics=False) -> tuple[list, list]:
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
        cv.solvePnPRefineLM(
            object_points,
            image_points,
            mtx,
            dist,
            rvec,
            tvec,
        )
        metrics = {}
        if getMetrics:
            projected, _ = cv.projectPoints(
                object_points,
                rvec,
                tvec,
                mtx,
                dist,
            )
            projected = projected.squeeze(1)

            distances = np.linalg.norm(projected - image_points, axis=1)

            metrics["AAE"] = np.mean(np.abs(distances))

            metrics["RSE"] = np.sqrt(np.mean(distances**2))

        return rvec, tvec, metrics
    else:
        raise Exception("Could not solve PnP")


def processArucoFromMultipleCameras(
    fixedArucos: list[Aruco],
    movingArucos: list[Aruco],
    cameras: list[Camera],
    frame: list[list],
    getMetrics=False,
):
    assert len(cameras) == len(frame)
    # This dict will contains each "variant" of the positions of each arucos corner. One for each time a camera see them
    arucosPositions: dict[int, list[np.ndarray[np.ndarray[float]]]] = {}
    for c, f in zip(cameras, frame):
        res = processAruco(
            fixedArucos,
            movingArucos,
            c,
            f,
            accept_none=True,
            directUpdate=False,
            metrics=getMetrics,
        )
        if not res:
            continue
        rvec, tvec, ap, met = res
        for key in ap.keys():
            if key not in arucosPositions:
                arucosPositions[key] = []
            arucosPositions[key].append(ap[key])


    for a in movingArucos:
        if a.id in arucosPositions:
            if getMetrics:
                nbrDims = 3  # We're in 3d
                nbrCorner = 4  # We have 4 corners for now

                arr = np.array(arucosPositions[a.id])
                # Get all unique pairs of indices
                indices = list(combinations(range(len(arr)), 2))

                # Compute differences for unique pairs
                diffs = np.array([arr[i] - arr[j] for i, j in indices])

                vals = np.square(diffs) / len(arr)  # val ** 2 / nbr_of_observed_values

                err = np.sum(
                    vals, axis=0
                )  # This gives us an error per coordinate per corner for this aruco marker

                err = np.sum(err, axis=0) / nbrCorner  # This gives us an error per axes

                RMSE_per_axes = np.sqrt(err)

                RMSE_total = np.sum(RMSE_per_axes) / nbrDims

            average_positions = np.mean(arucosPositions[a.id], axis=0)
            for i, newPos in enumerate(average_positions):
                a.corners[i].updatePos(newPos)