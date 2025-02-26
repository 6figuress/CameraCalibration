from time import sleep
import numpy as np
import cv2 as cv
from aruco import *
from pygame.time import Clock
import open3d as o3d


class Camera:
    deviceId: int
    captureStream: cv.VideoCapture
    mtx: np.ndarray
    dist: np.ndarray
    position: np.ndarray
    rotation: np.ndarray

    def __init__(self, deviceId, calibrationFile):
        self.deviceId = deviceId
        self.calibrationFile = calibrationFile
        self.captureStream = cv.VideoCapture(deviceId)
        # self.captureStream.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        # self.captureStream.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        self.mtx, self.dist = loadCalibration(calibrationFile)


cameras = [
    # Camera(0, "calibration/integrated_full.npz"),
    Camera(2, "calibration/logitec_2_f30.npz"),
    Camera(4, "calibration/logitec_4_f30.npz"),
]


def markMarkers(frame):
    """
    Mark the detected markers on the frame

    Args:
        frame (np.ndarray): Image frame

    Returns:
        np.ndarray: Image frame with markers marked
    """
    arucos, rejected = detectAruco(frame)
    arucos = list(arucos.values())
    for i in range(0, len(arucos)):
        arucos[i] = arucos[i][0]

    drawArucoCorners(frame, arucos)
    return frame


def locateCamera(cameras: list[Camera], arucos: dict[int, Aruco]):
    """
    Visualize the cameras and arucos in 3D space

    Args:
        cameras (list[Camera]): List of Camera objects
        arucos (dict[int, Aruco]): Dictionary of Aruco objects

    Returns:
        None
    """
    cube_size = 3.0
    colors = {
        0: [0.0, 0.0, 1.0],
        1: [0.0, 1.0, 0.0],
        2: [1.0, 0.0, 0.0],
        3: [1.0, 1.0, 0.0],
    }
    meshs = []
    for a in arucos.values():
        # Create a TriangleMesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(a.getCornersAsList())
        mesh.triangles = o3d.utility.Vector3iVector(
            [[0, 1, 2], [0, 2, 3]]
        )  # Two triangles
        # Optional: Add color
        mesh.paint_uniform_color(colors[a.id])  # Light blue
        # Compute normals for better lighting
        mesh.compute_vertex_normals()
        meshs.append(mesh)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for m in meshs:
        # Visualize
        vis.add_geometry(m)

    cubes = []

    for c in cameras:
        # Cube representing camera
        cube = o3d.geometry.TriangleMesh.create_box(cube_size, cube_size, cube_size)
        cube.paint_uniform_color([0.8, 0.5, 0.5])  # Purple
        cube.compute_vertex_normals()
        cube.translate([15, 15, 0])
        cubes.append(cube)
        vis.add_geometry(cube)

    # Setting open3d view position
    ctl = vis.get_view_control()
    ctl.set_constant_z_far(1000)
    ctl.camera_local_translate(forward=-100, right=0.0, up=0)
    while True:
        for i, camera in enumerate(cameras):
            ret, frame = camera.captureStream.read()
            if not ret:
                continue
            rvec, tvec = getPosition(
                frame, camera.mtx, camera.dist, arucos, accept_none=True
            )
            if rvec is None:
                continue
            R, _ = cv.Rodrigues(rvec)
            camera_world_position = -R.T @ tvec
            camera.position = camera_world_position
            camera.rotation = R
            cubes[i].translate(camera_world_position, relative=False)

            vis.update_geometry(cubes[i])
        for m in meshs:
            vis.update_geometry(m)
        vis.poll_events()
        vis.update_renderer()


arucos: dict[int, Aruco] = {
    0: Aruco(0, Position(0, 0, 0), size=3),
    1: Aruco(1, Position(15, 3, 0), size=3),
    2: Aruco(2, Position(3, 18, 0), size=3),
    3: Aruco(3, Position(18, 19, 0), size=3),
}

locateCamera(cameras, arucos)

cv.destroyAllWindows()
