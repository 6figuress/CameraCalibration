from time import sleep
import numpy as np
import cv2 as cv
from referential import Transform
from aruco import Aruco, getArucosFromPaper, processAruco
from scipy.spatial.transform import Rotation as R
from position import Position
from camera import Camera
from plotting import vizPoses
import mathutils
import bpy

arucos: dict[int, Aruco] = getArucosFromPaper()


def renderFromCamera(camera: Camera, filepath: str = "./blender/base.blend"):
    bpy.ops.wm.open_mainfile(filepath=filepath)

    base_transf = camera.world2cam.invert

    cam_blender = Transform(tvec=base_transf.tvec * 0.001, rot_mat=base_transf.rot_mat)

    cam_pose = cam_blender.apply(np.array([0, 0, 0]))

    bpy.ops.object.camera_add(location=cam_pose)
    cam = bpy.context.object

    bpy.context.scene.camera = cam
    # camera properties
    bpy.data.scenes["Scene"].camera.data.angle_x = 1.2217305
    bpy.data.scenes["Scene"].camera.data.angle_y = 0.7504916
    bpy.data.scenes["Scene"].camera.data.lens_unit = "FOV"

    bpy.data.scenes["Scene"].camera.rotation_mode = "QUATERNION"
    bpy.data.scenes["Scene"].camera.rotation_quaternion = camera.world2cam.quat

    bpy.ops.object.light_add(type="POINT", location=(0, 0, 5))

    # Set render resolution (optional)
    bpy.context.scene.render.resolution_x = camera._resolution[0]
    bpy.context.scene.render.resolution_y = camera._resolution[1]

    # Set the render output path
    bpy.context.scene.render.filepath = "./blender/rendered_image.png"

    # Render the image
    bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath="./blender/test.blend")


logi_a = Camera("Logitec_A", 4, focus=0, resolution=(1280, 720))

sleep(1)

frame = logi_a.takePic()

cv.imwrite("./blender/base.png", frame)


rvec, tvec, _, _ = processAruco(arucos.values(), [], logi_a, frame)

renderFromCamera(logi_a)
