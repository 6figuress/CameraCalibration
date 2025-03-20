[![codecov](https://codecov.io/gh/6figuress/CameraCalibration/branch/main/graph/badge.svg?token=W25GRHWTS0)](https://codecov.io/gh/6figuress/CameraCalibration)

# Camera syncronization

This repository is used to calibrate a camera, and then use it to locate itself from aruco markers used as "anchors". And then located other aruco marker in a defined 3d coordinate space

## To install

To install the package, you can either install it via git directly using `pip install git+https://github.com/6figuress/CameraCalibration.git`

Or you can clone the repository in local, and install it via `pip install -e <path to module>`. This method allow you to modifiy the module and test it directly

## Tweaks

If you're using wayland as a render server, you may need to run `export XDG_SESSION_TYPE=x11` for it to work
