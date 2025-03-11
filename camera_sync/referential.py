import numpy as np
import cv2 as cv
from cv2.typing import MatLike, Vec3f
from typing import Self


class Transform:
    _transf_mat: MatLike = None
    _rvec: Vec3f = None
    _inv: Self = None

    def __init__(
        self,
        transf_mat: MatLike = None,
        rvec: Vec3f = None,
        tvec: Vec3f = None,
        rot_mat: MatLike = None,
    ):
        """
        Parameters:
            transf_mat: The complete transformation matrix
            rvec: A compressed Rodrigues vector
            tvec: A translation vect
            rot_mat: A rotation matrix (3x3)

        ## Notes:
            You need to give either :
            - The transformation matrix
            - The translation and rotation vector
            - The translation vector and rotation matrix
        """
        if rvec is not None:
            if rot_mat is not None:
                raise Exception(
                    "Give either the rotation vector or the rotation matrix, but not both"
                )
            self._rvec = rvec
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
        if self._rvec is None:
            self._rvec, _ = cv.Rodrigues(self.rot_mat)
        return self._rvec

    @property
    def tvec(self) -> Vec3f:
        return self.transf_mat[:3, 3]

    @property
    def invert(self) -> Self:
        """
        Invert a transformation

        Returns:
            Transform: A new transformation that is the inverse of this one
        """
        if self._inv is None:
            self._inv = Transform(
                rot_mat=self.rot_mat.T, tvec=-self.rot_mat.T @ self.tvec
            )
        return self._inv

    def apply(self, point: Vec3f) -> Vec3f:
        """
        Apply the transformation to a 3d point

        Returns:
            Vec3f: The new point
        """
        return self.rot_mat @ point + self.tvec

    def combine(self, t2: Self) -> Self:
        """
        Combines this transform with another transform by matrix multiplication.
        This operation is equivalent to applying the other transform after this one.
        The resulting transform represents the sequential application of the two transforms.
        Parameters:
            t2 (Transform): The second transform to combine with this one
        Returns:
            Transform: A new transformation that is the combination of this transform followed by t

        ## Notes:
            This is equivalent to **t2 @ self**. This mean that the transformation passed in parameter (t2) is applied **after** this one !
        """

        return Transform(transf_mat=t2.transf_mat @ self.transf_mat)
