from setuptools import find_packages, setup

setup(
    name="camera_sync",
    version="0.1",
    packages=find_packages(),
    install_requires=["opencv-python", "matplotlib", "scipy",  "bpy", "numpy<2"],
    author="6 Figures",
    description="A package to allow camera to localize aruco markers in a 3d world",
)
