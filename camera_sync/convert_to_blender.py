import numpy as np
import bpy

from aruco import Aruco
from position import Position

arucos: dict[int, Aruco] = {
    10: Aruco(10, size=80, topLeft=Position(10, 15, 0)),
    11: Aruco(11, size=80, topLeft=Position(120, 15, 0)),
    # 12: Aruco(12, size=80, topLeft=Position(10, 110, 0)),
    13: Aruco(13, size=80, topLeft=Position(120, 110, 0)),
    # 14: Aruco(14, size=80, topLeft=Position(10, 205, 0)),
    # 15: Aruco(15, size=80, topLeft=Position(120, 205, 0)),
}


def convertToBlender(arcuo: Aruco):
    center = arcuo.getCenter().coords
    return np.array([center[0], -center[1], center[2]])


bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

for a in arucos.values():
    pos = convertToBlender(a)
    bpy.ops.mesh.primitive_plane_add(
        size=a.size / 1000,
        enter_editmode=False,
        location=(pos[0] / 1000, pos[1] / 1000, pos[2] / 1000),
    )

# Add a camera
bpy.ops.object.camera_add(location=(5, -5, 5))
camera = bpy.context.object

camera.rotation_euler = (1.1093, 0, 0.7854)  # Adjust the rotation if needed

bpy.context.scene.camera = camera

bpy.ops.object.light_add(type="POINT", location=(0, 0, 5))

# Set render resolution (optional)
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080

# Set the render output path
bpy.context.scene.render.filepath = "./rendered_image.png"

# Render the image
bpy.ops.render.render(write_still=True)
