from .position import Point, Position, lowPassFilter, ALPHA
import pytest
import numpy as np


def test_init_point():
    test = Point(0.56, 2)

    assert isinstance(test.x, float)
    assert isinstance(test.y, float)
    assert test.y != 0.56
    assert test.x != 2.0
    assert test.x == 0.56
    assert test.y == 2.0


def test_point_properties():
    point = Point(1.5, 2.5)
    assert point.x == 1.5
    assert point.y == 2.5
    assert np.array_equal(point.coords, np.array([1.5, 2.5]))


def test_position_init():
    pos = Position(1.0, 2.0, 3.0)
    assert pos.x == 1.0
    assert pos.y == 2.0
    assert pos.z == 3.0
    assert np.array_equal(pos.coords, np.array([1.0, 2.0, 3.0]))


def test_low_pass_filter():
    new_pos = np.array([10.0, 20.0])
    old_pos = np.array([5.0, 15.0])
    alpha = 0.4
    
    filtered = lowPassFilter(new_pos, old_pos, alpha)
    expected = np.array([7.0, 17.0])  # 0.4 * new + 0.6 * old
    
    assert np.array_equal(filtered, expected)


def test_point_update_pos_no_filter():
    point = Point(1.0, 2.0)
    new_pos = np.array([5.0, 6.0])
    
    point.updatePos(new_pos, noFilter=True)
    assert np.array_equal(point.coords, new_pos)


def test_point_update_pos_with_filter():
    point = Point(5.0, 15.0)
    new_pos = np.array([10.0, 20.0])
    old_pos = np.array(point.coords)
    point.updatePos(new_pos, noFilter=False)
    xChange = new_pos[0] - old_pos[0]
    yChange = new_pos[1] - old_pos[1]
    print(xChange, old_pos[0])
    expected = np.array(
        [old_pos[0] + xChange * ALPHA, old_pos[1] + yChange * ALPHA]
    )  # Using default ALPHA=0.4

    assert np.array_equal(point.coords, expected)


def test_position_inheritance():
    pos = Position(1.0, 2.0, 3.0)
    
    # Test that Position inherits from Point
    assert isinstance(pos, Point)
    
    # Test that Position extends Point
    assert hasattr(pos, 'z')
    assert pos.z == 3.0
