from camera import Camera
import time
import cv2
from ultralytics import YOLO

cam = Camera("192.168.50.222", 8899, "admin", "admin123")



while True:
    
    cam.move(pan=0.5)
    time.sleep(2)
    cam.stop()
    
    cam.move(tilt=-0.5)
    time.sleep(1)
    cam.stop()

    cam.move(pan=-0.5)
    time.sleep(2)
    cam.stop()

    cam.move(tilt=0.5)
    time.sleep(1)
    cam.stop()