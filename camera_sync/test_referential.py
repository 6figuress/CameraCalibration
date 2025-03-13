import pytest
import numpy as np
from .referential import Transform


def test_init_with_transf_mat():
    """Test initialization with transformation matrix"""
    transf_mat = np.eye(4)
    transform = Transform(transf_mat=transf_mat)
    assert np.array_equal(transform.transf_mat, transf_mat)

def test_init_with_rvec_tvec():
    """Test initialization with rotation vector and translation vector"""
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([1.0, 2.0, 3.0])
    transform = Transform.fromRodrigues(rvec=rvec, tvec=tvec)
    
    # Check if tvec is correctly stored
    assert np.allclose(transform.tvec, tvec)
    # Check if rvec is correctly stored
    print("Rvec is : ", transform.rvec)
    print("Rvec is ", rvec)
    assert np.allclose(transform.rvec, rvec)


def test_init_with_rot_mat_tvec():
    """Test initialization with rotation matrix and translation vector"""
    rot_mat = np.eye(3)
    tvec = np.array([1.0, 2.0, 3.0])
    transform = Transform.fromRotationMatrix(rot_mat=rot_mat, tvec=tvec)
    
    # Check if transformation matrix is correctly constructed
    expected_transf_mat = np.eye(4)
    expected_transf_mat[:3, 3] = tvec
    assert np.array_equal(transform.transf_mat, expected_transf_mat)

def test_properties():
    """Test properties of Transform class"""
    # Create a transform with known values
    rot_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    tvec = np.array([1.0, 2.0, 3.0])
    transform = Transform.fromRotationMatrix(rot_mat=rot_mat, tvec=tvec)
    
    # Test rot_mat property
    assert np.array_equal(transform.rot_mat, rot_mat)
    
    # Test tvec property
    assert np.array_equal(transform.tvec, tvec)
    
    # Test transf_mat property
    expected_transf_mat = np.eye(4)
    expected_transf_mat[:3, :3] = rot_mat
    expected_transf_mat[:3, 3] = tvec
    assert np.array_equal(transform.transf_mat, expected_transf_mat)


def test_rvec_property():
    """Test rvec property calculation when not provided during init"""
    # Create rotation matrix for 90 degrees around z-axis
    rot_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transform = Transform.fromRotationMatrix(rot_mat=rot_mat, tvec=[0, 0, 0])
    
    # Expected rvec for this rotation (approximation)
    # For 90 deg rotation around z-axis, expect something close to [0, 0, π/2]
    rvec = transform.rvec.flatten()
    assert abs(rvec[2] - np.pi/2) < 0.0001
    assert abs(rvec[0]) < 0.0001
    assert abs(rvec[1]) < 0.0001


def test_invert_property():
    """Test the invert property"""
    # Create a transform
    rot_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    tvec = np.array([1.0, 2.0, 3.0])
    transform = Transform.fromRotationMatrix(rot_mat=rot_mat, tvec=tvec)
    
    # Get the inverse
    inverse = transform.invert
    
    # Check inverse rotation is transpose of original
    assert np.array_equal(inverse.rot_mat, rot_mat.T)
    
    # Check inverse translation is -R^T * t
    expected_tvec = -rot_mat.T @ tvec
    assert np.array_equal(inverse.tvec, expected_tvec)

    inverted_matrix = np.linalg.inv(transform.transf_mat)
    assert np.array_equal(inverted_matrix, inverse.transf_mat)


def test_apply_method():
    """Test the apply method"""
    # Create a simple transform (90-degree rotation around Z + translation)
    rot_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    tvec = np.array([1.0, 2.0, 3.0])
    transform = Transform.fromRotationMatrix(rot_mat=rot_mat, tvec=tvec)
    
    # Apply to a point
    point = np.array([1.0, 0.0, 0.0])
    transformed_point = transform.apply(point)
    
    # Expected: rotation maps [1,0,0] to [0,1,0], then add translation
    expected_point = np.array([0.0, 1.0, 0.0]) + tvec
    assert np.allclose(transformed_point, expected_point)


def test_combine_method():
    """Test the combine method"""
    # Create two transformations
    t1 = Transform.fromRotationMatrix(
        rot_mat=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), tvec=np.array([1, 2, 3])
    )
    t2 = Transform.fromRotationMatrix(
        rot_mat=np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), tvec=np.array([4, 5, 6])
    )
    
    # Combine them
    combined = t1.combine(t2)
    
    # Check the combined transformation matrix
    expected_mat = t2.transf_mat @ t1.transf_mat
    assert np.array_equal(combined.transf_mat, expected_mat)
    
    # Test the combination by applying transformations in sequence to a point
    point = np.array([1.0, 1.0, 1.0])
    expected_point = t2.apply(t1.apply(point))
    actual_point = combined.apply(point)
    assert np.allclose(actual_point, expected_point)


def test_rotation_transform_correctness():
    """Test correctness of rotation transforms with well-known rotations"""
    # Test rotation around X axis by 90 degrees
    rx90 = Transform.fromRotationMatrix(
        rot_mat=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), tvec=[0, 0, 0]
    )

    point = np.array([1, 2, 3])
    transformed_point = rx90.apply(point)
    assert np.allclose(transformed_point, [1, -3, 2])

    # Test rotation around Y axis by 90 degrees
    ry90 = Transform.fromRotationMatrix(
        rot_mat=np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), tvec=[0, 0, 0]
    )

    transformed_point = ry90.apply(point)
    assert np.allclose(transformed_point, [3, 2, -1])

    # Test rotation around Z axis by 90 degrees
    rz90 = Transform.fromRotationMatrix(
        rot_mat=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), tvec=[0, 0, 0]
    )

    transformed_point = rz90.apply(point)
    assert np.allclose(transformed_point, [-2, 1, 3])


def test_transformation_chain_correctness():
    """Test correctness of a chain of transformations"""
    # Create a point
    point = np.array([1, 1, 1])

    # Create transformations - rotation + translation
    t1 = Transform.fromRotationMatrix(
        rot_mat=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), tvec=[5, 0, 0]
    )  # 90° Z rotation + X translation

    t2 = Transform.fromRotationMatrix(
        rot_mat=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), tvec=[0, 3, 0]
    )  # 90° X rotation + Y translation

    # Apply transformations sequentially
    p1 = t1.apply(point)  # Should be [5-1, 1+0, 1+0] = [4, 1, 1]
    p2 = t2.apply(p1)  # Should be [4+0, 3-1, 1+0] = [4, 2, 1]

    # Verify sequential application
    assert np.allclose(p1, [4, 1, 1])
    assert np.allclose(p2, [4, 2, 1])

    # Apply combined transformation
    combined = t1.combine(t2)
    transformed_point = combined.apply(point)

    # Results should match sequential application
    assert np.allclose(transformed_point, p2)


def test_inverse_transformation_correctness():
    """Test that applying a transform and then its inverse returns to the original point"""
    # Create different points to test
    points = [np.array([1, 0, 0]), np.array([0, 5, -2]), np.array([3.5, -2.7, 1.2])]

    # Create a complex transformation
    rot = np.array([0.3, -0.5, 0.7])  # Arbitrary rotation vector
    tvec = np.array([4.2, -1.3, 2.8])  # Arbitrary translation

    transform = Transform.fromRodrigues(rvec=rot, tvec=tvec)
    inverse = transform.invert

    # Test each point
    for point in points:
        # Transform point and then apply inverse
        transformed = transform.apply(point)
        restored = inverse.apply(transformed)

        # Original and restored points should be equal
        assert np.allclose(point, restored, atol=1e-10)


def test_quaternion_representation():
    """Test that quaternion representation is correct"""
    # Create a simple rotation - 180 degrees around Y axis
    # This should give a quaternion of (0, 0, 1, 0) in scalar-first format
    rot_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    transform = Transform.fromRotationMatrix(rot_mat=rot_mat, tvec=[0, 0, 0])

    # Check quaternion (scalar-first format)
    expected_quat = np.array([0, 0, 1, 0])
    assert np.allclose(transform.quat, expected_quat)

    # Create another rotation - 90 degrees around Z axis
    # Quaternion should be approximately (sqrt(2)/2, 0, 0, sqrt(2)/2) in scalar-first format
    rot_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transform = Transform.fromRotationMatrix(rot_mat=rot_mat, tvec=[0, 0, 0])

    expected_quat = np.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
    assert np.allclose(transform.quat, expected_quat)