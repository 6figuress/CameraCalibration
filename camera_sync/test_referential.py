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
    transform = Transform(rvec=rvec, tvec=tvec)
    
    # Check if tvec is correctly stored
    assert np.allclose(transform.tvec, tvec)
    # Check if rvec is correctly stored
    assert np.allclose(transform.rvec, rvec)


def test_init_with_rot_mat_tvec():
    """Test initialization with rotation matrix and translation vector"""
    rot_mat = np.eye(3)
    tvec = np.array([1.0, 2.0, 3.0])
    transform = Transform(rot_mat=rot_mat, tvec=tvec)
    
    # Check if transformation matrix is correctly constructed
    expected_transf_mat = np.eye(4)
    expected_transf_mat[:3, 3] = tvec
    assert np.array_equal(transform.transf_mat, expected_transf_mat)


def test_init_error_cases():
    """Test initialization error cases"""
    # Test when both rvec and rot_mat are provided
    with pytest.raises(Exception) as excinfo:
        Transform(rvec=[0.1, 0.2, 0.3], rot_mat=np.eye(3))
    assert "Give either the rotation vector or the rotation matrix" in str(excinfo.value)
    
    # Test when not enough information is given
    with pytest.raises(Exception) as excinfo:
        Transform()
    assert "Couldn't create transform, not enough information given" in str(excinfo.value)


def test_properties():
    """Test properties of Transform class"""
    # Create a transform with known values
    rot_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    tvec = np.array([1.0, 2.0, 3.0])
    transform = Transform(rot_mat=rot_mat, tvec=tvec)
    
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
    transform = Transform(rot_mat=rot_mat, tvec=[0, 0, 0])
    
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
    transform = Transform(rot_mat=rot_mat, tvec=tvec)
    
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
    transform = Transform(rot_mat=rot_mat, tvec=tvec)
    
    # Apply to a point
    point = np.array([1.0, 0.0, 0.0])
    transformed_point = transform.apply(point)
    
    # Expected: rotation maps [1,0,0] to [0,1,0], then add translation
    expected_point = np.array([0.0, 1.0, 0.0]) + tvec
    assert np.allclose(transformed_point, expected_point)


def test_combine_method():
    """Test the combine method"""
    # Create two transformations
    t1 = Transform(rot_mat=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), tvec=np.array([1, 2, 3]))
    t2 = Transform(rot_mat=np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), tvec=np.array([4, 5, 6]))
    
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