import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def loadCalibration(path):
    file = np.load(path)
    return file["mtx"], file["dist"]


def detectAruco(img):
    arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

    arucoParams = cv.aruco.DetectorParameters()
    (corners, ids, rejected) = cv.aruco.detectMarkers(
        img, arucoDict, parameters=arucoParams
    )
    cv.aruco.drawDetectedMarkers(img, corners, ids)
    cv.imwrite("test.png", img)
    return corners, ids, rejected


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


# We could try to make that work with 4 corners for each aruco marker
object_points = np.array(
    [
        (0.0, 0.0, 0.0),  # Top-left corner
        (190.0, 0.0, 0.0),  # Top-right corner
        (0.0, 190.0, 0.0),  # Bottom-left corner
        (190.0, 190.0, 0.0),  # Bottom-right corner
    ],
    dtype=np.float32,
)


def plot_camera_pose(positions, object_points):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot camera position
    for i, p in enumerate(positions):
        ax.scatter(*p.flatten(), color="black", label=f"Camera Position {i}")

    ax.plot(
        [object_points[0][0], object_points[1][0]],
        [object_points[0][1], object_points[1][1]],
        [object_points[0][2], object_points[1][2]],
        color="purple",
    )
    ax.plot(
        [object_points[1][0], object_points[3][0]],
        [object_points[1][1], object_points[3][1]],
        [object_points[1][2], object_points[3][2]],
        color="purple",
    )
    ax.plot(
        [object_points[3][0], object_points[2][0]],
        [object_points[3][1], object_points[2][1]],
        [object_points[3][2], object_points[2][2]],
        color="purple",
    )
    ax.plot(
        [object_points[2][0], object_points[0][0]],
        [object_points[2][1], object_points[0][1]],
        [object_points[2][2], object_points[0][2]],
        color="purple",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xbound(-1000, 1000)
    ax.set_ybound(-1000, 1000)
    ax.set_zbound(-1000, 1000)
    ax.legend()
    plt.show()


def getPosition(picPath, mtx, dist):
    img = cv.imread(picPath)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = detectAruco(gray)
    img_points = [i[0][0] for i in corners]
    rvec, tvec = PnP(img_points, mtx, dist, object_points)
    return rvec, tvec


mtx, dist = loadCalibration("./calibration/calibration.npz")
positions = []
for i in range(3):
    rvec, tvec = getPosition(f"./loca/{i}.jpg", mtx, dist)
    R, _ = cv.Rodrigues(rvec)  # Convert rotation vector to 3x3 rotation matrix
    camera_position = -R.T @ tvec  # Compute camera position in world coordinates
    camera_position[2] = abs(camera_position[2])
    print("Camera position in world coordinates:", camera_position.ravel())

    positions.append(camera_position)

    # plot_camera_pose([camera_position], object_points)


plot_camera_pose(positions, object_points)
# Example usage
