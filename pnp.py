import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def loadCalibration(path):
    file = np.load(path)
    return file["mtx"], file["dist"]


def detectAruco(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

    arucoParams = cv.aruco.DetectorParameters()
    (corners, ids, rejected) = cv.aruco.detectMarkers(
        gray, arucoDict, parameters=arucoParams
    )
    orderedCorners = [[] for i in range(len(ids))]
    for i, id in enumerate(ids):
        orderedCorners[id[0]] = corners[i]

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

    cv.imwrite("test.png", img)
    return orderedCorners, ids, rejected


def PnP(points, mtx, dist, object_points):
    # Define 3D points of the object (e.g., a square with known dimensions)

    image_points = np.array(points, dtype=np.float32)

    # Solve PnP to estimate rotation and translation

    # success, rvec, tvec = cv.solvePnP(object_points, image_points, mtx, dist)
    success, rvec, tvec, inliers = cv.solvePnPRansac(
        object_points, image_points, mtx, dist
    )
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


linking = [[0, 1], [1, 3], [3, 2], [2, 0]]

# We could try to make that work with 4 corners for each aruco marker
object_points = np.array(
    [
        # x,  y,  z
        # Top left aruco
        (0.0, 0.0, 0.0),  # top left
        # (1.0, 0.0, 0.0),  # top right
        # (0.0, 1.0, 0.0),  # bottom left
        # (1.0, 1.0, 0.0),  # bottom right
        # Top-right aruco
        (15.0, 3.0, 0.0),  # top left
        # (16.0, 3.0, 0.0),  # top right
        # (15.0, 4.0, 0.0),  # bottom left
        # (16.0, 4.0, 0.0),  # bottom right
        # Bottom-left aruco
        (3.0, 18.0, 0.0),  # top left
        # (4.0, 18.0, 0.0),  # top right
        # (3.0, 19.0, 0.0),  # bottom left
        # (4.0, 19.0, 0.0),  # bottom right
        # Bottom-right aruco
        (18.0, 19.0, 0.0),  # top left
        # (19.0, 19.0, 0.0),  # top right
        # (18.0, 20.0, 0.0),  # bottom left
        # (19.0, 20.0, 0.0),  # bottom right
    ],
    dtype=np.float32,
)


def plot_camera_pose(positions, rotations, object_points):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot camera positions
    for i, p in enumerate(positions):
        ax.scatter(*p.flatten(), color="black", label=f"Camera Position {i}")
        camera_direction = rotations[i][:, 2]
        endpoint = p.flatten() + 10 * camera_direction
        ax.plot(
            [p[0][0], endpoint[0]],
            [p[1][0], endpoint[1]],
            [p[2][0], endpoint[2]],
            color="purple",
        )

    for i, p in enumerate(object_points):
        ax.scatter(*p, label=f"Aruco {i}")

    # Plot object points (= aruco markers corners)
    for l in linking:
        ax.plot(
            [object_points[l[0]][0], object_points[l[1]][0]],
            [object_points[l[0]][1], object_points[l[1]][1]],
            [object_points[l[0]][2], object_points[l[1]][2]],
            color="red",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xbound(-50, 50)
    ax.set_ybound(-50, 50)
    ax.set_zbound(-50, 50)
    ax.legend()
    plt.show()


def getPosition(picPath, mtx, dist):
    img = cv.imread(picPath)
    corners, ids, rejected = detectAruco(img)
    img_points = []
    for c in corners:
        # Here c contains the 4 corners of a detected aruco marker
        img_points.append(c[0][0].tolist())
        # for p in c[0]:
        #     img_points.append([i for i in p])
    rvec, tvec = PnP(img_points, mtx, dist, object_points)
    return rvec, tvec


mtx, dist = loadCalibration("./calibration/calibration.npz")
positions = []
rotations = []
for i in range(2):
    rvec, tvec = getPosition(f"./loca/{i}.jpg", mtx, dist)
    R, _ = cv.Rodrigues(rvec)  # Convert rotation vector to 3x3 rotation matrix
    rotations.append(R)
    camera_position = -R.T @ tvec  # Compute camera position in world coordinates
    # camera_position[2] = abs(camera_position[2])
    print("Camera position in world coordinates:", camera_position.ravel())

    positions.append(camera_position)
    # plot_camera_pose([camera_position], object_points)


plot_camera_pose(positions, rotations, object_points)
# Example usage
