from setuptools import find_packages, setup

setup(
    name="camera_sync",
    version="0.1",
    packages=find_packages(),
    install_requires=["camera_sync", "opencv-python", "matplotlib", "open3d"],
    python_requires=">=3.11,<3.12",
    author="6 Figures",
    description="A package to allow camera to localize aruco markers in a 3d world",
)
