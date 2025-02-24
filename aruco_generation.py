import cv2
import cv2.aruco as aruco
import numpy as np


def generate_aruco_marker(
    dictionary_id=aruco.DICT_4X4_50,
    marker_id=0,
    marker_size=200,
    save_path="aruco_marker.png",
):
    """
    Generates an ArUco marker and saves it as an image.

    :param dictionary_id: The ArUco dictionary to use (default: DICT_4X4_50)
    :param marker_id: The ID of the marker to generate (should be within the dictionary range)
    :param marker_size: The size of the output image in pixels
    :param save_path: The file path to save the marker image
    """
    # Load the predefined dictionary
    aruco_dict = aruco.getPredefinedDictionary(dictionary_id)

    # Create an empty image to store the marker
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)

    # Generate the marker
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # Save the marker image
    cv2.imwrite(save_path, marker_image)
    print(f"ArUco marker (ID: {marker_id}) saved as {save_path}")


ids = range(0, 0)


for id in ids:
    # Example usage
    generate_aruco_marker(marker_id=id, save_path=f"./aruco/aruco_{id}.png")
