from setuptools import find_packages, setup

setup(
    name="camera_sync",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "camera_sync",
        "opencv-python",
        "opencv-contrib-python",
        "matplotlib",
        "open3d",
        "scipy",
        "bpy",
        "numpy",
    ],
    python_requires=">=3.10,<3.12",
    author="6 Figures",
    description="A package to allow camera to localize aruco markers in a 3d world",
    package_data={
        "camera_sync": [
            "calibration/*",
            "blender/base.blend",
            "blender/10_to_15.png",
        ],
    },
)
