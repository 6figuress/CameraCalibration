# Camera syncronization

This repository is used to calibrate a camera, and then use it to locate itself from aruco markers used as "anchors". And then located other aruco marker in a defined 3d coordinate space

## To install

 To install the package and run it directly, you can run `pip install -e .` in this folder. If you want to use this module in another python file, you can target the git folder with the install command, i.e. replace *.* with the path to this folder

## Tweaks

If you're using wayland as a render server, you may need to `source ./setup.sh` in order use open3D for vizualisation. This is necessary if you see warnings in the console about some open3d windows that failed to be created