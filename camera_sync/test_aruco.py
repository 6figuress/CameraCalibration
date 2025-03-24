import os
from .aruco import Aruco, processAruco
import pytest
import numpy as np
import cv2 as cv
from .aruco import Aruco, getArucosFromPaper, generate_aruco_marker, detectAruco
from .position import Position
from .camera import Camera
from .tools import getBaseFolder


class TestAruco:
    def test_init_without_topleft(self):
        # Test initialization without topLeft
        aruco = Aruco(1, 50)
        assert aruco.id == 1
        assert aruco.size == 50
        assert aruco.corners is None
        assert aruco.isLocated is False

    def test_init_with_topleft(self):
        # Test initialization with topLeft
        topleft = Position(10, 20, 30)
        aruco = Aruco(2, 50, topleft)
        assert aruco.id == 2
        assert aruco.size == 50
        assert aruco.isLocated is True
        assert len(aruco.corners) == 4
        # Check first corner (top left)
        assert aruco.corners[0].x == 10
        assert aruco.corners[0].y == 20
        assert aruco.corners[0].z == 30
        # Check second corner (top right)
        assert aruco.corners[1].x == 60  # 10 + 50
        assert aruco.corners[1].y == 20
        assert aruco.corners[1].z == 30
        # Check third corner (bottom right)
        assert aruco.corners[2].x == 60  # 10 + 50
        assert aruco.corners[2].y == -30  # 20 - 50
        assert aruco.corners[2].z == 30
        # Check fourth corner (bottom left)
        assert aruco.corners[3].x == 10
        assert aruco.corners[3].y == -30  # 20 - 50
        assert aruco.corners[3].z == 30

    def test_get_corners_from_topleft(self):
        # Test static method for generating corners
        topleft = np.array([10, 20, 30], dtype=float)
        size = 40
        corners = Aruco.getCornersFromTopLeft(topleft, size)

        assert len(corners) == 4
        # Check top left
        assert np.array_equal(corners[0], np.array([10, 20, 30]))
        # Check top right
        assert np.array_equal(corners[1], np.array([50, 20, 30]))  # 10 + 40
        # Check bottom right
        assert np.array_equal(corners[2], np.array([50, -20, 30]))  # 20 - 40
        # Check bottom left
        assert np.array_equal(corners[3], np.array([10, -20, 30]))  # 20 - 40

    def test_is_located_property(self):
        # Test without corners
        aruco1 = Aruco(1, 50)
        assert aruco1.isLocated is False

        # Test with corners
        aruco2 = Aruco(2, 50, Position(0, 0, 0))
        assert aruco2.isLocated is True

    def test_get_center(self):
        # Create an Aruco marker at a known position
        topleft = Position(10, 20, 30)
        size = 40
        aruco = Aruco(1, size, topleft)

        # Get center
        center = aruco.getCenter()

        # Center should be at (x + size/2, y - size/2, z)
        assert center.x == topleft.x + size / 2
        assert center.y == topleft.y - size / 2
        assert center.z == topleft.z

    def test_get_corners_as_list(self):
        # Create an Aruco marker
        topleft = Position(10, 20, 30)
        size = 40
        aruco = Aruco(1, size, topleft)

        # Get corners as list
        corners_list = aruco.getCornersAsList()

        # Verify format and values
        assert isinstance(corners_list, np.ndarray)
        assert corners_list.shape == (4, 3)  # 4 corners, 3 coordinates each

        # Check first corner
        assert np.array_equal(corners_list[0], np.array([10, 20, 30]))


class TestGetArucosFromPaper:
    def test_get_arucos_from_paper_version_1(self):
        # Test version 1
        arucos = getArucosFromPaper(1)

        # Check returned dictionary
        assert isinstance(arucos, dict)
        assert len(arucos) == 10  # 10 markers defined for version 1

        # Check specific IDs
        assert 0 in arucos
        assert 1 in arucos
        assert 10 in arucos
        assert 15 in arucos

        # Check a specific marker's properties
        assert arucos[11].id == 11
        assert arucos[11].size == 80
        assert arucos[11].corners[0].x == 110  # topLeft position
        assert arucos[11].corners[0].y == 0
        assert arucos[11].corners[0].z == 0

    def test_get_arucos_from_paper_version_2(self):
        # Test version 2
        arucos = getArucosFromPaper(2)

        # Check returned dictionary
        assert isinstance(arucos, dict)
        assert len(arucos) == 25  # 5x5 grid of markers

        # Check specific IDs in expected range
        for id in range(20, 45):
            assert id in arucos
            assert arucos[id].size == 30

        # Check that grid pattern is maintained
        # First row, first column
        assert arucos[20].corners[0].x == 0
        assert arucos[20].corners[0].y == 0

        # First row, second column
        assert arucos[21].corners[0].x == 35
        assert arucos[21].corners[0].y == 0

        # Second row, first column
        assert arucos[25].corners[0].x == 0
        assert arucos[25].corners[0].y == -35

    def test_get_arucos_from_paper_version_3(self):
        # Test version 3
        arucos = getArucosFromPaper(3)

        # Check returned dictionary
        assert isinstance(arucos, dict)
        assert len(arucos) == 4  # 4 markers defined for version 3

        # Check specific IDs
        assert 44 in arucos
        assert 45 in arucos
        assert 46 in arucos
        assert 47 in arucos

        # Check properties of markers
        assert arucos[44].size == 77
        assert arucos[44].corners[0].x == 0
        assert arucos[44].corners[0].y == 0

        assert arucos[47].corners[0].x == 150.4
        assert arucos[47].corners[0].y == -220

    def test_default_parameter(self):
        # Test default parameter (pap_v=1)
        arucos_default = getArucosFromPaper()
        arucos_explicit = getArucosFromPaper(1)

        # Check they are the same
        assert len(arucos_default) == len(arucos_explicit)
        for id in arucos_default:
            assert id in arucos_explicit
            assert arucos_default[id].size == arucos_explicit[id].size


@pytest.mark.parametrize("marker_id", [0, 10, 20])
def test_generate_aruco_marker(marker_id, monkeypatch):
    # Mock cv.imwrite and set up expectation that it will be called
    def mock_imwrite(path, img):
        assert path == f"./aruco/aruco_{marker_id}.png"
        return True

    monkeypatch.setattr(cv, "imwrite", mock_imwrite)

    # Mock print function to avoid output during tests
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    # Call function with mocked dependencies
    result = generate_aruco_marker(marker_id)

    # Verify result is a numpy array
    assert isinstance(result, np.ndarray)


def test_detect_aruco():
    frame1 = cv.imread(os.path.join(getBaseFolder(), "test_material/pic_1.png"))
    frame2 = cv.imread(os.path.join(getBaseFolder(), "test_material/pic_2.png"))

    res1 = detectAruco(frame1)[0]
    res2 = detectAruco(frame2)[0]

    print(res1)

    must_detect = range(10, 16)

    frame1_pos = {
        10: np.array(
            [
                [391.99924, 271.00888],
                [296.54086, 273.53796],
                [295.7735, 209.53203],
                [382.2614, 207.31253],
            ]
        ),
        11: np.array(
            [
                [260.21378, 274.4671],
                [163.56339, 277.09277],
                [175.34366, 212.59235],
                [262.5721, 210.44418],
            ]
        ),
        12: np.array(
            [
                [380.4164, 196.12245],
                [295.69894, 198.22845],
                [295.09512, 147.1313],
                [372.60196, 145.03049],
            ]
        ),
        13: np.array(
            [
                [263.1719, 198.9502],
                [177.41513, 201.00327],
                [186.79846, 149.55522],
                [265.03305, 147.83983],
            ]
        ),
        14: np.array(
            [
                [371.28998, 135.70331],
                [294.81262, 137.67108],
                [294.38434, 95.868996],
                [364.77438, 94.0064],
            ]
        ),
        15: np.array(
            [
                [265.35855, 138.41458],
                [188.45963, 140.09364],
                [195.93839, 97.7269],
                [266.87744, 96.51508],
            ]
        ),
    }

    frame2_pos = {
        10: np.array(
            [
                [708.62256, 329.4364],
                [577.3158, 328.6003],
                [583.8624, 246.86192],
                [704.0657, 247.42998],
            ]
        ),
        11: np.array(
            [
                [527.4651, 328.55966],
                [396.12662, 327.64578],
                [418.5065, 246.26654],
                [538.2975, 246.9151],
            ]
        ),
        12: np.array(
            [
                [703.23175, 232.63849],
                [585.1846, 232.45404],
                [590.52185, 166.09595],
                [699.26154, 166.04037],
            ]
        ),
        13: np.array(
            [
                [540.1074, 232.38245],
                [422.4393, 231.67505],
                [440.5373, 165.51047],
                [548.7781, 166.07718],
            ]
        ),
        14: np.array(
            [
                [698.408, 153.57715],
                [591.4622, 153.79189],
                [595.8921, 98.87692],
                [695.2108, 98.71832],
            ]
        ),
        15: np.array(
            [
                [550.4101, 153.69283],
                [443.68304, 153.33621],
                [458.6222, 98.24213],
                [557.5898, 98.89872],
            ]
        ),
    }

    for n in must_detect:
        assert n in res1.keys()
        assert n in res2.keys()

    for k in res1.keys():
        assert np.allclose(res1[k], frame1_pos[k], atol=1e-3)

    for k in res2.keys():
        assert np.allclose(res2[k], frame2_pos[k], atol=1e-3)

    pass


def test_process_aruco():
    cam = Camera("Logitec_A", -1, focus=0, resolution=(1280, 720))
    frame = cv.imread(os.path.join(getBaseFolder(), "test_material/pic_2.png"))
    anchors = [10, 11, 14, 15]
    moving = [12, 13]
    arucos = getArucosFromPaper(1)

    achorsMarkers = []
    movingMarkers = []

    for n in moving:
        movingMarkers.append(arucos[n])

    for n in anchors:
        achorsMarkers.append(arucos[n])

    realRvec = np.array([0.00686839, -2.92023025, 1.16051479])
    realTvec = np.array([37.15601032, -37.2392403, 580.35313275])

    rvec, tvec, found, metrics = processAruco(achorsMarkers, movingMarkers, cam, frame, metrics=True)

    assert metrics["PnP"]["MAE"] < 0.6
    assert metrics["PnP"]["RMSE"] < 0.8
    assert np.allclose(realRvec, rvec.flatten(), atol=1e-4)
    assert np.allclose(realTvec, tvec.flatten(), atol=1e-4)


    for n in moving:
        assert n in found

    pass