import cv2 as cv
from .camera import Camera


cam = Camera("Logitec_A", 2, focus=0, resolution=(1280, 720))

frame  =cam.takePic()
cv.imwrite("test.png", frame)