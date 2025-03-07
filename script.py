import bpy

def pos_cam(loc, rot, camera, scene):

    camera.location.x = loc[0]
    camera.location.y = loc[1]
    camera.location.z = loc[2]

    scene.camera.rotation_mode = 'XYZ'
    scene.camera.rotation_euler[0] = rot[0]
    scene.camera.rotation_euler[1] = rot[1]
    scene.camera.rotation_euler[2] = rot[2]

    # camera properties
    bpy.data.scenes["Scene"].camera.data.angle_x = 1.2217305
    bpy.data.scenes["Scene"].camera.data.angle_y = 0.7504916
    bpy.data.scenes["Scene"].camera.data.lens_unit = "FOV"

    # Set the render output path
    bpy.context.scene.render.filepath = "//camera_view.png"

    # Render and save the image
    bpy.ops.render.render(write_still=True)
    print("Rendered image saved!")


obj = bpy.data.objects['Camera']
scene = bpy.data.scenes["Scene"]

scene.render.resolution_x = 639
scene.render.resolution_y = 478


x = -0.2669273290066621
y = -0.1446455231733305
z = -0.3794566648480968

loc = [x, y, z]

rotation_x = -0.61492632
rotation_y = -1.02643127
rotation_z = -1.97325078

rot = [rotation_x, rotation_y, rotation_z]

pos_cam(loc, rot, obj, scene)