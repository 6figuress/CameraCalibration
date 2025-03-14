import os
from time import sleep
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import bpy

from .tools import getBaseFolder
from .referential import Transform
from .aruco import Aruco, getArucosFromPaper, processAruco
from .camera import Camera

arucos: dict[int, Aruco] = getArucosFromPaper()

BLENDER_FOLDER = os.path.join(getBaseFolder(), "./blender")


def renderFromCamera(
    camera: Camera,
    filepath: str = os.path.join(BLENDER_FOLDER, "base.blend"),
):
    bpy.ops.wm.open_mainfile(filepath=filepath)

    base_transf = camera.world2cam.invert

    cam_blender = Transform.fromRotationMatrix(
        tvec=base_transf.tvec * 0.001, rot_mat=base_transf.rot_mat
    )

    cam_pose = cam_blender.apply(np.array([0, 0, 0]))

    bpy.ops.object.camera_add(location=cam_pose)

    bpy.context.scene.camera = camera_setup(camera)

    # Apply 180edg X-axis rotation to align OpenCV's coordinate system (Z forward, Y down, X right) 
    # with Blender's coordinate system (Z forward, Y up, X right)
    cv_to_blender = R.from_euler("x", 180, degrees=True).as_matrix()
    blend_rot_mat = base_transf.rot_mat @ cv_to_blender 
    blend_quat = R.from_matrix(blend_rot_mat).as_quat(scalar_first=True)

    bpy.data.scenes["Scene"].camera.rotation_mode = "QUATERNION"
    bpy.data.scenes["Scene"].camera.rotation_quaternion = blend_quat 

    bpy.ops.object.light_add(type="POINT", location=(0, 0, 5))

    # Set render resolution (optional)
    bpy.context.scene.render.resolution_x = camera._resolution[0]
    bpy.context.scene.render.resolution_y = camera._resolution[1]

    # Set the render output path
    bpy.context.scene.render.filepath = os.path.join(
        BLENDER_FOLDER, "rendered_image.png"
    )

    # Render the image
    bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(BLENDER_FOLDER, "test.blend"))


def camera_setup(cam: Camera) -> bpy.types.Object:
    """Sets parameters of the Blender camera to match the characteristics of the real camera as closely as possible."""
    blender_cam = bpy.context.object
    scene = bpy.context.scene

    width = cam._resolution[0]
    height = cam._resolution[1]

    fx = cam.mtx[0, 0] # focal length in x direction in pixels 
    fy = cam.mtx[1, 1] # focal length in y direction in pixels
    cx = cam.mtx[0, 2] # principal point x
    cy = cam.mtx[1, 2] # principal point y

    # Hardcoded values for the camera sensor size - works for Logitech C920
    # Need to find out how to compute these values from the camera matrix
    blender_cam.data.angle_x = 1.2217305
    blender_cam.data.angle_y = 0.7504916
    blender_cam.data.lens_unit = "FOV"

    # Set the camera's principal point to match the camera matrix's principal point
    # https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/
    blender_cam.data.shift_x = -(cx / width - 0.5)
    blender_cam.data.shift_y = (cy - 0.5 * height) / width 

    pixel_aspect = fy / fx
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = pixel_aspect

    apply_distortion(cam)

    return blender_cam


def apply_distortion(cam: Camera) -> None:
    """Setups a NodeCompositorNodeMovieDistortion node to apply radial distortion to the rendered image."""

    width = cam._resolution[0]
    height = cam._resolution[1]

    # Get radial distortion values
    dist = cam.dist[0]
    k1 = float(dist[0])
    k2 = float(dist[1])
    k3 = float(dist[4])
    
    # Need to create a temporary image to load into Blender
    temp_img_path = os.path.join(os.getcwd(), "temp_img.png") 
    blank_img = np.zeros((height, width, 3), dtype=np.uint8)
    cv.imwrite(temp_img_path, blank_img)

    clip = bpy.data.movieclips.load(temp_img_path)
    clip.name = "Distortion" 
    
    # Distortion parameters need to be set on the movie clip's tracking camera
    # Blender only seems to support radial distortion 
    tracking_camera = clip.tracking.camera
    tracking_camera.k1 = k1
    tracking_camera.k2 = k2
    tracking_camera.k3 = k3

    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links

    # Clear existing nodes, we're starting from scratch
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    render_layer = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    
    movie_distortion = tree.nodes.new("CompositorNodeMovieDistortion")
    movie_distortion.distortion_type = "DISTORT"
    movie_distortion.clip = clip
    
    links.new(render_layer.outputs["Image"], movie_distortion.inputs["Image"])
    links.new(movie_distortion.outputs["Image"], composite.inputs["Image"])


if __name__ == "__main__":
    logi_a = Camera("Logitec_A", 2, focus=0, resolution=(1280, 720))

    sleep(1)

    frame = logi_a.takePic()

    cv.imwrite(os.path.join(BLENDER_FOLDER, "base.png"), frame)

    rvec, tvec, _, _ = processAruco(arucos.values(), [], logi_a, frame)

    renderFromCamera(logi_a)
