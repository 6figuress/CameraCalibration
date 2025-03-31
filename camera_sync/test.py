import cv2

try:
    # Try to access the aruco module
    cv2.aruco
    print("ArUco module is available")
except AttributeError:
    print("ArUco module is NOT available")
