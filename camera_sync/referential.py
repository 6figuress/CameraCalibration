import numpy as np
import cv2 as cv
from cv2.typing import MatLike, Vec3f, Vec4f
from typing import Self
from scipy.spatial.transform import Rotation as R


class Transform:
    _transf_mat: MatLike = None
    _rvec: Vec3f = None
    _inv: Self = None
    _quat: Vec4f = None

    @staticmethod
    def fromRotationMatrix(rot_mat: MatLike, tvec: Vec3f) -> Self:
        """
        Create a transformation from a rotation matrix and a translation vector

        Parameters:
            rot_mat: The rotation matrix
            tvec: The translation vector
        Returns:
            Transform: The transformation
        """

        tvec = np.array(tvec).flatten()

        transf_mat = np.eye(4)

        transf_mat[:3, :3] = rot_mat
        transf_mat[:3, 3] = tvec

        return Transform(transf_mat=transf_mat)

    @staticmethod
    def fromQuaternion(quat: Vec4f, tvec: Vec3f, scalar_first=True) -> Self:
        """
        Creates a Transform object from a quaternion and translation vector.

        Parameters:
            quat :
                The quaternion representing rotation. If scalar_first is True, the format is [w, x, y, z],
                otherwise [x, y, z, w].
            tvec :
                The translation vector [x, y, z].
            scalar_first:
                Whether the quaternion has the scalar component first.
        Returns:
        --------
        Self
            A new Transform object representing the specified transformation.
        """

        rot_mat = R.from_quat(quat, scalar_first=scalar_first).as_matrix()

        return Transform.fromRotationMatrix(rot_mat=rot_mat, tvec=tvec)

    @staticmethod
    def fromRodrigues(rvec: Vec3f, tvec: Vec3f) -> Self:
        """
        Create a transformation from a Rodrigues vector and a translation vector

        Parameters:
            rvec: The Rodrigues vector
            tvec: The translation vector
        Returns:
            Transform: The transformation
        """

        rot_mat = cv.Rodrigues(np.array(rvec).flatten())[0]

        return Transform.fromRotationMatrix(rot_mat=rot_mat, tvec=tvec)

    def __init__(self, transf_mat: MatLike = None):
        """
        Parameters:
            transf_mat: The complete transformation matrix
        """
        self._transf_mat = transf_mat
        self._preCompute()

    def _preCompute(self):
        self._rvec = cv.Rodrigues(self.rot_mat)[0].flatten()
        self._quat = R.from_matrix(self.rot_mat).as_quat(scalar_first=True)

    @property
    def transf_mat(self) -> MatLike:
        return self._transf_mat

    @property
    def quat(self) -> Vec4f:
        if self._quat is None:
            self._quat = R.from_matrix(self.rot_mat).as_quat(scalar_first=True)
        return self._quat

    @property
    def rot_mat(self) -> MatLike:
        return self.transf_mat[:3, :3]

    @property
    def rvec(self) -> Vec3f:
        return self._rvec

    @property
    def tvec(self) -> Vec3f:
        return self.transf_mat[:3, 3]

    @property
    def kine_pose(self):
        """
        Return the current transformation into a format compatible with the inverse kinematics

        Returns:
            np.array: The transformation in the format [x, y, z, w, x, y, z] with units : [m, m, m, rad, rad, rad]
        """
        temp = np.roll(self.quat, -1)
        return np.array([*self.tvec / 1000, *temp])

    @property
    def invert(self) -> Self:
        """
        Invert a transformation

        Returns:
            Transform: A new transformation that is the inverse of this one
        """
        if self._inv is None:
            self._inv = Transform.fromRotationMatrix(
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
