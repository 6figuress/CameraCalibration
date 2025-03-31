import cv2 as cv
import numpy as np
import open3d as o3d

from .aruco import (
    Aruco,
    getArucosFromPaper,
    processArucoFromMultipleCameras,
)
from .camera import Camera


def applyToMeshes(anchorMeshes, movingMeshes, camerasCubes, func):
    for m in movingMeshes.values():
        for c in m:
            func(c)

    for m in list(anchorMeshes.values()) + camerasCubes:
        func(m)


def initVisualization(
    cameras: list[Camera], baseMarkers: dict[int, Aruco], movingMarker: dict[int, Aruco]
):
    """
    Initialize the visualization window, create all the object to represent cameras and markers

    Args:
        cameras (list[Camera]): List of Camera objects
        baseMarkers (dict[int, Aruco]): Dictionary of Aruco objects that are anchors in the world coordinate system
        movingMarker (dict[int, Aruco]): Dictionary of Aruco objects that are moving in the world coordinate system

    Returns:
        tuple: Tuple containing the visualization object, the base markers meshes, the moving markers meshes and the cameras cubes
    """
    camera_size = 30
    camera_color = [0, 0, 1]
    corner_color = [0, 1, 0]
    corner_size = 10
    baseMarkersMeshes = {}
    movingMarkersMeshes = {}

    initialPosition = [0, 0, 0]

    def createMesh(corners, color=[1.0, 0.0, 0.0]):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(corners)
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        return mesh

    def createCube(pos, size, color):
        cube = o3d.geometry.TriangleMesh.create_box(size, size, size)
        cube.paint_uniform_color(color)  # Purple
        cube.compute_vertex_normals()
        cube.translate(pos)
        return cube

    for a in baseMarkers.values():
        baseMarkersMeshes[a.id] = createMesh(a.getCornersAsList())

    for a in movingMarker.values():
        movingMarkersMeshes[a.id] = [
            createCube(initialPosition, corner_size, corner_color) for i in range(4)
        ]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    camerasCube = []

    for c in cameras:
        camerasCube.append(createCube(initialPosition, camera_size, camera_color))

    applyToMeshes(
        baseMarkersMeshes,
        movingMarkersMeshes,
        camerasCube,
        lambda x: vis.add_geometry(x),
    )

    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True

    ctl = vis.get_view_control()
    ctl.set_constant_z_far(1000000)
    ctl.camera_local_translate(forward=-1000, right=40, up=-80)

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
        fixedMarkers (dict[int, Aruco]): Dictionary of Aruco objects that are anchors in the world coordinate system
        movingMarkers (dict[int, Aruco]): Dictionary of Aruco objects that are moving in the world coordinate system
    Returns:
        None
    """

    vis, baseMarkersMeshes, movingMarkersMeshes, camerasCube = initVisualization(
        cameras, fixedMarkers, movingMarkers
    )

    while True:
        frames = []
        for c in cameras:
            ret, frame = c.captureStream.read()
            if not ret:
                raise Exception("Aha ?")
            frames.append(frame)
        processArucoFromMultipleCameras(
            fixedMarkers.values(), movingMarkers, cameras, frames
        )

        for m in movingMarkers.values():
            for j, c in enumerate(movingMarkersMeshes[m.id]):
                c.translate(m.corners[j].coords, relative=False)

        for i, c in enumerate(cameras):
            camerasCube[i].translate(c.world_position.coords, relative=False)

        applyToMeshes(
            baseMarkersMeshes,
            movingMarkersMeshes,
            camerasCube,
            lambda x: vis.update_geometry(x),
        )
        vis.poll_events()
        vis.update_renderer()


if __name__ == "__main__":
    cameras = [
        Camera(
            "Logitec_A",
            2,
            focus=0,
            resolution=(1280, 720),
        ),
        # Camera(
        #     "Logitec_B",
        #     4,
        #     focus=0,
        #     resolution=(1280, 720),
        # ),
    ]

    arucos = getArucosFromPaper(1)

    achors = [10, 11, 14, 15]
    moving = [12, 13]
    fixedMarkers = {}

    movingMarkers = {}

    for i in achors:
        fixedMarkers[i] = arucos[i]

    for i in moving:
        movingMarkers[i] = arucos[i]

    locateCamera(cameras, fixedMarkers, movingMarkers)

    cv.destroyAllWindows()
