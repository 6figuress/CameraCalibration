# Camera syncronization

This repository is used to calibrate a camera, and then use it to locate itself from aruco markers used as "anchors". And then located other aruco marker in a defined 3d coordinate space

## To install

 To install the package and run it directly, you can run `pip install -e <absolute path to folder>` in your python environment.

 If you don't want to use the pre-calbrated files, you can use `pip install git+https://github.com/6figuress/CameraCalibration.git@feature/package` instead

## Tweaks

If you're using wayland as a render server, you may need to run `export XDG_SESSION_TYPE=x11` for it to work