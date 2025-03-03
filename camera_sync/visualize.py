import cv2 as cv
import open3d as o3d
from aruco import Aruco, Camera, Position, processAruco


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
    cube_size = 30
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

    for a in baseMarkers.values():
        baseMarkersMeshes[a.id] = createMesh(a.getCornersAsList())

    for a in movingMarker.values():
        movingMarkersMeshes[a.id] = createMesh(a.getCornersAsList(), [0.0, 1.0, 0.0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    camerasCube = []

    for c in cameras:
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
        vis.add_geometry(m)

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
        fixedMarkers (dict[int, Aruco]): Dictionary of Aruco objects that are anchors in the world coordinate system
        movingMarkers (dict[int, Aruco]): Dictionary of Aruco objects that are moving in the world coordinate system
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

            if not res:
                # Camera position could not be found
                continue

            locatedArucos = res[2]

            if i == 0:
                for id in locatedArucos.keys():
                    # This is a hack to position the aruco marker almost correctly
                    # TODO: Fix this by finding a nice way to move the marker (maybe tvec and rvec ?)
                    pos = locatedArucos[id].corners[0].coords
                    pos[0] += locatedArucos[id].size / 2
                    pos[1] -= locatedArucos[id].size / 2
                    movingMarkersMeshes[id].translate(pos, relative=False)

            camerasCube[i].translate(camera.world_position, relative=False)

        for m in (
            list(baseMarkersMeshes.values())
            + list(movingMarkersMeshes.values())
            + camerasCube
        ):
            vis.update_geometry(m)
        vis.poll_events()
        vis.update_renderer()


if __name__ == "__main__":
    cameras = [
        # Camera(0, "calibration/integrated_full.npz"),
        Camera(2, "calibration/logitec_2_f30.npz"),
        Camera(4, "calibration/logitec_4_f30.npz"),
    ]

    # Positions are in mm
    # Those dimensions and positions match the ones present in the inkscape file located in ./inkscape/10_to_15.svg
    # The origin is then placed in the top left corner of the paper
    arucos: dict[int, Aruco] = {
        10: Aruco(10, size=80, topLeft=Position(10, 15, 0)),
        11: Aruco(11, size=80, topLeft=Position(120, 15, 0)),
        12: Aruco(12, size=80, topLeft=Position(10, 110, 0)),
        13: Aruco(13, size=80, topLeft=Position(120, 110, 0)),
        14: Aruco(14, size=80, topLeft=Position(10, 205, 0)),
        15: Aruco(15, size=80, topLeft=Position(120, 205, 0)),
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
