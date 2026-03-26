import cv2

rtsp_url = "rtsp://admin:admin123@192.168.50.222:554/live/ch00_0"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

cv2.namedWindow("UDP Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("UDP Stream", 640, 360)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("UDP Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()