import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
rtsp_url = "rtsp://192.168.50.222:554/live/ch00_0"
cap = cv2.VideoCapture(rtsp_url)

cv2.namedWindow("YOLO Person Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Person Detection", 640, 360)  

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True)

        for result in results:
            annotated_frame = result.plot()

            cv2.imshow("YOLO Person Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()