import os
import math
import numpy as np
from camera import Camera
import cv2 as cv
from referential import Transform
from camera_sync.aruco import getArucosFromPaper, processAruco
from position import invertRefChange
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def getPosesPics():
    folderPath = "./pics/poses"

    pics = []

    for i, f in enumerate(sorted(os.listdir(folderPath))):
        pics.append(cv.imread(os.path.join(folderPath, f)))

    return pics


def debug_mat(transforms: list[Transform]):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    length = 50  # Shorter for cleaner visualization

    # Define all three directions to show full orientation
    directions = [
        [length, 0, 0],  # X-axis
        [0, length, 0],  # Y-axis
        [0, 0, length],  # Z-axis
    ]

    colors = [
        "r",  # X-axis in red
        "g",  # Y-axis in green
        "b",  # Z-axis in blue
    ]

    for i, t in enumerate(transforms):
        ax.scatter(*t.tvec, s=100, label=f"Position {i}")

        # Draw all three axes
        for d, c in zip(directions, colors):
            newPoint = t.apply(d)
            ax.plot(
                [t.tvec[0], newPoint[0]],
                [t.tvec[1], newPoint[1]],
                [t.tvec[2], newPoint[2]],
                color=c,
                linewidth=2,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Adjust limits to see all points
    ax.set_xlim((-500, 500))
    ax.set_ylim((-500, 100))
    ax.set_zlim((0, 500))

    ax.legend()
    plt.title("Robot End Effector Poses")
    plt.tight_layout()
    import ipdb

    ipdb.set_trace()
    plt.show()


arucos = getArucosFromPaper()


pics = getPosesPics()

robot_cam = Camera("robot", -1, focus=500)


camera_world_rvecs = []
camera_world_tvecs = []


# # Those are data from Sebastien - Not corresponding to any pictures
# translation_data = [
#     # [-93.14, -291.73, -210.34],
#     # [-53.87, -349.20, -193.23],
#     # [-110.15, -348.84, -192.95],
#     # [-54.15, -297.68, -193.44],
#     # [-104.76, -298.16, -193.99],
#     # Real world data
#     [-244.32, -128.5, -119.5],
#     [46.43, -217.58, -83.53],
#     [-385.26, -79.67, -178.37],
#     [-208.78, -437.98, -58.25],
#     [224.5, -337.9, -217.5],
#     [-260.7, 69.5, -140.6],
#     [-472.5, -22.6, -268.0],
#     [334.8, -235.2, -210.9],
# ]


# rotation_data = [
#     # [2.193, 2.209, 0.012],
#     # [2.409, 2.017, 0.000],
#     # [2.409, 2.017, 0.0],
#     # [2.399, 2.029, 0.000],
#     # [2.399, 2.028, -0.000],
#     # Real world data,
#     [3.720, -0.023, 0.099],
#     [2.000, -1.975, -0.091],
#     [2.990, 0.704, 0.006],
#     [0.008, -2.904, 0.724],
#     [1.739, -2.103, -0.176],
#     [2.844, 0.169, 0.018],
#     [2.345, 1.133, -0.401],
#     [2.075, -1.724, -0.071],
# ]

robot_poses = []

robot_rot_matrices = []

# for i, r in enumerate(rotation_data):
#     rot_mat, _ = cv.Rodrigues(np.array(r))
#     robot_poses.append(
#         Transform(
#             tvec=translation_data[i],
#             rot_mat=rot_mat,
#         )
#     )


robot_poses = np.array(
    [
        Transform(tvec=[244.32, 128.5, 340.5], rvec=[3.720, -0.023, 0.099]),
        Transform(tvec=[-46.43, 217.58, 376.47], rvec=[2.000, -1.975, -0.091]),
        Transform(tvec=[385.26, 79.67, 281.63], rvec=[2.990, 0.704, 0.006]),
        Transform(tvec=[208.78, 437.98, 401.75], rvec=[0.008, -2.904, 0.724]),
        Transform(tvec=[-224.5, 337.9, 242.5], rvec=[1.739, -2.103, -0.176]),
        Transform(tvec=[260.7, -69.5, 319.4], rvec=[2.844, 0.169, 0.018]),
        Transform(tvec=[472.5, 22.6, 192.0], rvec=[2.345, 1.133, -0.401]),
        Transform(tvec=[-334.8, 235.2, 249.1], rvec=[2.075, -1.724, -0.071]),
    ]
)


debug_mat(robot_poses)

t_base2gripper = np.array([t.tvec for t in robot_poses])

r_base2gripper = np.array([t.rot_mat for t in robot_poses])

camera_poses: list[Transform] = []

for p in pics:
    rvec, tvec, ar_pos, metrics = processAruco(arucos.values(), [], robot_cam, p)
    camera_poses.append(Transform(rvec=rvec, tvec=tvec))
    camera_world_rvecs.append(rvec)
    camera_world_tvecs.append(tvec)

rot_base2world, tvec_base2world, rot_grip2cam, tvec_grip2cam = (
    cv.calibrateRobotWorldHandEye(
        camera_world_rvecs,
        camera_world_tvecs,
        r_base2gripper,
        t_base2gripper,
        method=cv.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    )
)


base2world = Transform(rot_mat=rot_base2world, tvec=tvec_base2world)
grip2cam = Transform(rot_mat=rot_grip2cam, tvec=tvec_grip2cam)


def testCalib(
    cam: Camera, world2base: Transform, base2grip: Transform, grip2cam: Transform
):
    cam2world = cam.world2cam.invert()

    point = np.array([0.0, 0.0, 0.0])

    currStep = world2base.apply(point)

    currStep = base2grip.apply(currStep)

    currStep = grip2cam.apply(currStep)

    currStep = cam2world.apply(currStep)

    import ipdb
    ipdb.set_trace()

testCalib(robot_cam, base2world.invert(), robot_poses[-1], grip2cam)