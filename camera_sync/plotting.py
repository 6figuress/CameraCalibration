import matplotlib.pyplot as plt
from aruco import Aruco


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
