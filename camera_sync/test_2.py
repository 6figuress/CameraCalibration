import os
import math
import numpy as np
from camera import Camera
import cv2 as cv
from referential import Transform
from camera_sync.aruco import getArucosFromPaper, processAruco
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def vizPoses(poses: list[Transform], axis=2):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    length = 100

    base_axis = np.zeros(3)

    base_axis[axis] = length

    # Iterate over each pair of rvec and tvec
    for i, p in enumerate(poses):
        world_axis = p.rot_mat.dot(base_axis)
        newPoint = p.apply(base_axis)
        # Convert rvec to rotation matrix
        tvec = p.tvec

        # Plot the camera position (tvec)
        ax.scatter(tvec[0], tvec[1], tvec[2], color="k", s=100)
        # ax.scatter(*newPoint, color="g")

        # x = [tvec[0], newPoint[0]]
        # y = [tvec[1], newPoint[1]]
        # z = [tvec[2], newPoint[2]]

        # ax.plot(x, y, z, color="g")

        # Plot the camera Z-axis direction (world frame)
        ax.quiver(
            tvec[0],
            tvec[1],
            tvec[2],
            world_axis[0],
            world_axis[1],
            world_axis[2],
            length=length,
            normalize=True,
            label=f"Camera Z-axis Direction {i}",
        )

    # Set labels and limits for clarity
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim((-500, 500))
    ax.set_ylim((-500, 500))
    ax.set_zlim((-500, 500))

    plt.show()


def getPosesPics():
    folderPath = "./pics/poses"

    pics = []

    for f in sorted(os.listdir(folderPath)):
        pics.append(cv.imread(os.path.join(folderPath, f)))

    return pics


arucos = getArucosFromPaper()

anchors = [10, 11, 12, 13, 14, 15]

achorsMarkers = []

for i in anchors:
    achorsMarkers.append(arucos[i])


pics = getPosesPics()

robot_cam = Camera("robot", -1, focus=500)


camera_world_rvecs = []
camera_world_tvecs = []


rotation_data = [
    [3.720, -0.023, 0.099],
    [2.000, -1.975, -0.091],
    [2.990, 0.704, 0.006],
    [0.008, -2.904, 0.724],
    [1.739, -2.103, -0.176],
    [2.844, 0.169, 0.018],
    [2.345, 1.133, -0.401],
    [2.075, -1.724, -0.071],
]
translation_data = [
    [-244.32, -128.5, -119.5],
    [46.43, -217.58, -83.53],
    [-385.26, -79.67, -178.37],
    [-208.78, -437.98, -58.25],
    [224.5, -337.9, -217.5],
    [-260.7, 69.5, -140.6],
    [-472.5, -22.6, -268.0],
    [334.8, -235.2, -210.9],
]


robot_poses = []

for i, r in enumerate(rotation_data):
    rot = R.from_euler("xyz", r)
    rot.as_matrix()
    robot_poses.append(Transform(tvec=translation_data[i], rot_mat=rot.as_matrix()))


vizPoses(robot_poses[0])


t_base2gripper = np.array([t.tvec for t in robot_poses])


# TODO: Check if conversion order is correct
r_base2gripper = np.array([t.rot_mat for t in robot_poses])

camera_poses: list[Transform] = []

for p in pics:
    rvec, tvec, ar_pos, metrics = processAruco(achorsMarkers, [], robot_cam, p)
    camera_poses.append(Transform(rvec=rvec, tvec=tvec))
    camera_world_rvecs.append(rvec)
    camera_world_tvecs.append(tvec)

camera_world_poses = [p.invert() for p in camera_poses]


rot_base2world, tvec_base2world, rot_grip2cam, tvec_grip2cam = (
    cv.calibrateRobotWorldHandEye(
        camera_world_rvecs, camera_world_tvecs, r_base2gripper, t_base2gripper
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

    pass


def checkCalibForPose(
    pic,
    cam: Camera,
    base2grip: Transform,
    world2base: Transform,
    grip2cam: Transform,
):
    points = np.array(
        [
            [0, 0, 0],
            [10, 15, 0],
            [120, 15, 0],
        ],
        dtype=np.float32,
    )

    cam.world2cam.invert().apply([0, 0, 0])

    cam.world2cam.apply([0, 0, 0])

    world_origin = [0, 0, 0]

    # Now we need to go from world perspective to camera
    img_points, _ = cv.projectPoints(points, cam.rvec, cam.tvec, cam.mtx, cam.dist)

    img_points = np.rint(img_points).astype(np.int32)

    img_points = img_points.squeeze(1)

    drawedPic = pic.copy()
    for p in img_points:
        print("Point pos : ", p)
        drawedPic = cv.circle(drawedPic, p, 2, (0, 255, 255))

    return drawedPic


testCalib(robot_cam, base2world.invert(), robot_poses[-1], grip2cam)

# newPic = checkCalibForPose(
#     pics[7], robot_cam, robot_poses[7], base2world.invert(), grip2cam
# # )
# cv.imwrite("frame.png", newPic)
# cv.waitKey(0)
