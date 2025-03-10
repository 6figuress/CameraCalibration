import numpy as np

ALPHA = 0.4


class Point:
    coords: np.ndarray[float]
    oldCoords: np.ndarray[float]

    def __init__(self, x: float, y: float):
        self.coords = np.array([x, y])

    @property
    def x(self) -> float:
        return self.coords[0]

    @property
    def y(self) -> float:
        return self.coords[1]

    def updatePos(self, newPos: np.ndarray[float], noFilter=False):
        if noFilter:
            self.coords = newPos
        else:
            self.coords = lowPassFilter(newPos, self.coords, ALPHA)
        # self.coords = newPos


class Position(Point):

    def __init__(self, x: float, y: float, z: float):
        self.coords = np.array([x, y, z])

    @property
    def z(self) -> float:
        return self.coords[2]


def refChange(position: np.ndarray, rot_mat, tvec):
    return rot_mat @ position + tvec


def invertRefChange(position: np.ndarray, rot_mat, tvec):
    inv_tvec = -rot_mat.T @ tvec
    return rot_mat.T @ position + inv_tvec


def lowPassFilter(
    new_position: np.ndarray[float], old_position: np.ndarray[float], alpha: float
) -> np.ndarray[float]:
    return alpha * new_position + (1 - alpha) * old_position