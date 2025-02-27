from time import sleep
import numpy as np
import cv2 as cv
from aruco import *
from pygame.time import Clock
import open3d as o3d


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


def locateCamera(cameras: list[Camera], arucos: list[int, Aruco]):
    """
    Visualize the cameras and arucos in 3D space

    Args:
        cameras (list[Camera]): List of Camera objects
        arucos (dict[int, Aruco]): Dictionary of Aruco objects

    Returns:
        None
    """
    cube_size = 30
    meshs = []
    for a in arucos.values():
        # Create a TriangleMesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(a.getCornersAsList())
        mesh.triangles = o3d.utility.Vector3iVector(
            [[0, 1, 2], [0, 2, 3]]
        )  # Two triangles
        # Optional: Add color
        mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Light blue
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
    ctl.set_constant_z_far(100000)
    ctl.camera_local_translate(forward=-1000, right=0.0, up=0)
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

            position, R = locateCameraWorld(rvec, tvec)

            camera.position = position
            camera.rotation = R
            print("Position is : ", camera.position)
            cubes[i].translate(camera.position, relative=False)

            vis.update_geometry(cubes[i])
        for m in meshs:
            vis.update_geometry(m)
        vis.poll_events()
        vis.update_renderer()


cameras = [
    # Camera(0, "calibration/integrated_full.npz"),
    Camera(0, "calibration/logitec_2_f30.npz"),
    # Camera(2, "calibration/logitec_4_f30.npz"),
]

# Positions are in mm
# Those dimensions and positions match the ones present in the inkscape file located in ./inkscape/10_to_15.svg
# The origin is then placed in the top left corner of the paper
arucos: dict[int, Aruco] = {
    10: Aruco(10, Position(10, 15, 0), size=80),
    11: Aruco(11, Position(120, 15, 0), size=80),
    12: Aruco(12, Position(10, 110, 0), size=80),
    13: Aruco(13, Position(120, 110, 0), size=80),
    14: Aruco(14, Position(10, 205, 0), size=80),
    15: Aruco(15, Position(120, 205, 0), size=80),
}


fixed = [10, 11, 14, 15]
toLocate = {}

for i in fixed:
    toLocate[i] = arucos[i]


locateCamera(cameras, toLocate)

cv.destroyAllWindows()
