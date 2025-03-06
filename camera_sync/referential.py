import numpy as np
import cv2 as cv
from cv2.typing import MatLike, Vec3f
from typing import Self


class Transform:
    _transf_mat: MatLike = None

    def __init__(
        self,
        transf_mat: MatLike = None,
        rvec: Vec3f = None,
        tvec: Vec3f = None,
        rot_mat: MatLike = None,
    ):
        if rvec is not None:
            rot_mat, _ = cv.Rodrigues(np.array(rvec))

        if tvec is not None and rot_mat is not None:
            self._transf_mat = np.eye(4)
            self._transf_mat[:3, :3] = rot_mat
            self._transf_mat[:3, 3] = np.array(tvec).ravel()
        elif transf_mat is not None:
            self._transf_mat = transf_mat
        else:
            raise Exception("Couldn't create transform, not enough information given.")

    @property
    def transf_mat(self) -> MatLike:
        return self._transf_mat

    @property
    def rot_mat(self) -> MatLike:
        return self.transf_mat[:3, :3]

    @property
    def rvec(self) -> Vec3f:
        rvec, _ = cv.Rodrigues(self.rot_mat)
        return rvec

    @property
    def tvec(self) -> Vec3f:
        return self.transf_mat[:3, 3]

    def apply(self, point: Vec3f) -> Vec3f:
        return point @ self.rot_mat + self.tvec

    def invert(self) -> Self:
        return Transform(transf_mat=np.linalg.inv(self.transf_mat))

    def combine(self, t: Self) -> Self:
        return Transform(transf_mat=self.transf_mat @ t.transf_mat)
