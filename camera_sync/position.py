import numpy as np

ALPHA = 0.4


class Point:
    coords: np.ndarray[float]

    def __init__(self, x: float, y: float):
        self.coords = np.array([x, y])

    @property
    def x(self) -> float:
        return self.coords[0]

    @property
    def y(self) -> float:
        return self.coords[1]

    def updatePos(self, newPos: np.ndarray[float], noFilter=False) -> None:
        """
        Updates the position coordinates of the object.

        Parameters:
            newPos : The new position coordinates to update.
            noFilter : If True, updates the position without applying a filter.

                        If False, applies a low-pass filter to the new position coordinates before updating.

                        Default is False.
        Returns:
            None
        """
        if noFilter:
            self.coords = newPos
        else:
            self.coords = lowPassFilter(newPos, self.coords, ALPHA)


class Position(Point):

    def __init__(self, x: float, y: float, z: float):
        self.coords = np.array([x, y, z])

    @property
    def z(self) -> float:
        return self.coords[2]


def lowPassFilter(
    new_position: np.ndarray[float], old_position: np.ndarray[float], alpha: float
) -> np.ndarray[float]:
    return alpha * new_position + (1 - alpha) * old_position