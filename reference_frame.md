# Transformation matrix

## What does that mean ?

So to pass from a reference frame to another, we can use a 4x4 matrix, that can contain a translation, a rotation and a scaling.

For example, to pass from the camera reference frame to the world, we can express a point in the camera frame, for example [0, 0, 0] (which is the origin in the camera frame, io the camera itself).

Then, to see where that point would be in the world reference frame, we need to apply the transformation.

#### The notation is the following : <sup>W</sup>T<sub>C</sub>

In text, this means W is the target frame (world in our case). And C is the source frame (camera in our case).

With this, P<sub>C</sub> = [0, 0, 0] is the origin in the camera frame

Then, P<sub>W</sub> = <sup>W</sup>T<sub>C</sub> **·** P<sub>C</sub>

So P<sub>W</sub> is the position of our camera in the world frame