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


def initVisualization(
    cameras: list[Camera], baseMarkers: dict[int, Aruco], movingMarker: dict[int, Aruco]
):
    cube_size = 30
    baseMarkersMeshes = {}
    movingMarkersMeshes = {}

    initialPosition = [0, 0, 0]

    def createMesh(corners, color=[1.0, 0.0, 0.0]):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(corners)
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        mesh.paint_uniform_color(color)  # Light blue
        # Compute normals for better lighting
        mesh.compute_vertex_normals()
        return mesh

    for a in baseMarkers.values():
        # Create a TriangleMesh
        baseMarkersMeshes[a.id] = createMesh(a.getCornersAsList())

    for a in movingMarker.values():
        movingMarkersMeshes[a.id] = createMesh(a.getCornersAsList(), [0.0, 1.0, 0.0])

        # movingMarkersMeshes[a.id] = createMesh(a.getCornersAsList(), [0.0, 1.0, 0.0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    camerasCube = []

    for c in cameras:
        # Cube representing camera
        cube = o3d.geometry.TriangleMesh.create_box(cube_size, cube_size, cube_size)
        cube.paint_uniform_color([0.8, 0.5, 0.5])  # Purple
        cube.compute_vertex_normals()
        cube.translate(initialPosition)
        camerasCube.append(cube)

    for m in (
        list(baseMarkersMeshes.values())
        + list(movingMarkersMeshes.values())
        + camerasCube
    ):
        # Visualize
        vis.add_geometry(m)

    # Setting open3d view position
    ctl = vis.get_view_control()
    ctl.set_constant_z_far(100000)
    ctl.camera_local_translate(forward=-1000, right=0.0, up=0)

    return vis, baseMarkersMeshes, movingMarkersMeshes, camerasCube


def locateCamera(
    cameras: list[Camera],
    fixedMarkers: dict[int, Aruco],
    movingMarkers: dict[int, Aruco],
):
    """
    Visualize the cameras and arucos in 3D space

    Args:
        cameras (list[Camera]): List of Camera objects
        arucos (dict[int, Aruco]): Dictionary of Aruco objects

    Returns:
        None
    """

    vis, baseMarkersMeshes, movingMarkersMeshes, camerasCube = initVisualization(
        cameras, fixedMarkers, movingMarkers
    )

    while True:
        for i, camera in enumerate(cameras):
            ret, frame = camera.captureStream.read()
            if not ret:
                continue

            res = processAruco(
                fixedMarkers.values(),
                movingMarkers.values(),
                camera,
                frame,
                accept_none=True,
            )

            if res == False:
                # Camera position could not be found
                continue

            locatedArucos = res[2]

            if i == 0:

                for id in locatedArucos.keys():
                    # Update the vertices of the mesh
                    pos = locatedArucos[id].corners[0].toList()
                    pos[0] += locatedArucos[id].size / 2
                    pos[1] -= locatedArucos[id].size / 2
                    movingMarkersMeshes[id].translate(pos, relative=False)

                print(movingMarkersMeshes[13])

            camerasCube[i].translate(camera.world_position, relative=False)

        for m in (
            list(baseMarkersMeshes.values())
            + list(movingMarkersMeshes.values())
            + camerasCube
        ):
            vis.update_geometry(m)
        vis.poll_events()
        vis.update_renderer()


cameras = [
    # Camera(0, "calibration/integrated_full.npz"),
    Camera(2, "calibration/logitec_2_f30.npz"),
    Camera(4, "calibration/logitec_4_f30.npz"),
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
moving = [12, 13]
fixedMarkers = {}

movingMarkers = {}

for i in fixed:
    fixedMarkers[i] = arucos[i]

for i in moving:
    movingMarkers[i] = arucos[i]


locateCamera(cameras, fixedMarkers, movingMarkers)

cv.destroyAllWindows()
