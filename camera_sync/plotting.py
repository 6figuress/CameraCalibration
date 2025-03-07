import matplotlib.pyplot as plt
import numpy as np
from aruco import Aruco
from referential import Transform


def vizPoses(poses: list[Transform], axis=2):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    length = 100

    base_axis = np.zeros(3)

    base_axis[axis] = length

    # Iterate over each pair of rvec and tvec
    for i, p in enumerate(poses):
        world_axis = p.rot_mat.dot(base_axis)
        newPoint = p.apply(base_axis)
        # Convert rvec to rotation matrix
        R = p.rot_mat
        tvec = p.tvec

        # Plot the camera position (tvec)
        ax.scatter(tvec[0], tvec[1], tvec[2], color="k", s=100)
        ax.scatter(*newPoint, color="g")

        x = [tvec[0], newPoint[0]]
        y = [tvec[1], newPoint[1]]
        z = [tvec[2], newPoint[2]]

        ax.plot(x, y, z, color="g")

        # Plot the camera Z-axis direction (world frame)
        ax.quiver(
            tvec[0],
            tvec[1],
            tvec[2],
            world_axis[0],
            world_axis[1],
            world_axis[2],
            length=length,
            normalize=True,
            label=f"Camera Z-axis Direction {i}",
        )

    # Set labels and limits for clarity
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim((-500, 500))
    ax.set_ylim((-500, 500))
    ax.set_zlim((-500, 500))

    plt.show()


def draw_camera_position(ax, position, rotation):
    """
    Draw the camera position and direction of looking in 3D.

    Args:
        ax: The 3D plot axis
        position: The camera position
        rotation: The camera rotation

    Returns:
        The point and line objects
    """
    point = ax.scatter(*position.flatten(), color="black")
    camera_direction = rotation[:, 2]
    endpoint = position.flatten() + 10 * camera_direction
    # Plot the camera direction of looking
    line = ax.plot(
        [position[0][0], endpoint[0]],
        [position[1][0], endpoint[1]],
        [position[2][0], endpoint[2]],
        color="purple",
    )
    return point, line


def init_3d_plot():
    """
    Initialize the 3d plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xbound(-50, 50)
    ax.set_ybound(-50, 50)
    ax.set_zbound(-50, 50)
    return ax, fig


def draw_arucos(ax, arucos: list[Aruco]):
    """
    Draw the arucos positions

    Args:
        ax: The graph axes
        arucos: The list of arucos to draw
    """
    for aruco in arucos:
        # Draw the aruco id on top of the marker
        ax.text(aruco.corners[0].x, aruco.corners[0].y - 1, 0.0, aruco.id, fontsize=20)
        for i in range(0, len(aruco.corners)):
            # Draw a square corresponding to the marker position
            ax.plot(
                [aruco.corners[i].x, aruco.corners[(i + 1) % 4].x],
                [aruco.corners[i].y, aruco.corners[(i + 1) % 4].y],
                [aruco.corners[i].z, aruco.corners[(i + 1) % 4].z],
                color="black",
            )


def plot_camera_pose(positions: list, rotations: list, arucos: list[Aruco]):
    """
    Plot the camera pose and object points in 3D.

    Args:
        positions: The camera positions
        rotations: The camera rotations
        arucos: The arucos
    """

    ax, fig = init_3d_plot()

    # Plot camera positions
    for i, p in enumerate(positions):
        ax.scatter(*p.flatten(), color="black", label=f"Camera Position {i}")
        camera_direction = rotations[i][:, 2]
        endpoint = p.flatten() + 10 * camera_direction
        # Plot the camera direction of looking
        ax.plot(
            [p[0][0], endpoint[0]],
            [p[1][0], endpoint[1]],
            [p[2][0], endpoint[2]],
            color="purple",
        )

    draw_arucos(ax, arucos)

    ax.legend()
    plt.show()
