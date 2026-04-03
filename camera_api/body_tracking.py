import cv2
from ultralytics import YOLO
from camera import Camera
import threading
import time

# ── Configuration ─────────────────────────────────────────
CAM_IP         = "192.168.1.11"
CAM_PORT       = 8899
CAM_USER       = "admin"
CAM_PASS       = "admin123"
RTSP_URL       = f"rtsp://{CAM_IP}:554/live/ch00_0"
MODEL_PATH     = "yolov8n.pt"

FRAME_W, FRAME_H = 1920, 1080
DEAD_ZONE      = 0.15
PAN_SPEED      = 0.5
TILT_SPEED     = 0.3
MOVE_DURATION  = 0.02
CONF_THRESHOLD = 0.5
# ──────────────────────────────────────────────────────────

cam   = Camera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(RTSP_URL)

cv2.namedWindow("YOLOv8 Person Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Person Detection", FRAME_W, FRAME_H)

cam_busy = False


def move_camera(pan: float, tilt: float):
    global cam_busy
    if cam_busy:
        return
    def _move():
        global cam_busy
        cam_busy = True
        cam.move(pan=pan, tilt=tilt)
        time.sleep(MOVE_DURATION)
        cam.stop()
        cam_busy = False
    threading.Thread(target=_move, daemon=True).start()


def best_detection(results, frame_w, frame_h):
    """Return bounding box of the largest valid person detection."""
    best_box  = None
    best_area = 0
    for result in results:
        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0 or conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            '''
            if bw > frame_w * 0.85 or bh > frame_h * 0.85:
                continue
            '''
            if area > best_area:
                best_area = area
                best_box  = (x1, y1, x2, y2)
    return best_box


def put_text_right(img, text, row, color=(255, 255, 255), font_scale=1.1, thickness=2):
    """Draw text anchored to the right side of the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = img.shape[1] - tw - 20       
    y = 50 + row * 45                 
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


if not cap.isOpened():
    print("Error: could not open camera stream.")
else:
    print("[INFO] Running YOLOv8 Person Detection. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed, retrying...")
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        results = list(model(frame, stream=True, conf=CONF_THRESHOLD))

        annotated = frame.copy()
        person_count = 0

        # Draw all detected persons
        for result in results:
            for box in result.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls != 0 or conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw = x2 - x1
                bh = y2 - y1

                person_count += 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated, f"Person {person_count} {conf:.2f}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # Track the largest/closest person with camera
        box = best_detection(results, w, h)

        cx, cy, pan, tilt = 0.0, 0.0, 0.0, 0.0

        if box is not None:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(annotated, "TRACKING",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cx = ((x1 + x2) / 2) / w - 0.5
            cy = ((y1 + y2) / 2) / h - 0.5

            if abs(cx) > DEAD_ZONE:
                pan = max(-1.0, min(1.0, cx * PAN_SPEED * 2.5))

            if abs(cy) > DEAD_ZONE:
                tilt = max(-1.0, min(1.0, -cy * TILT_SPEED * 2.5))

            if pan != 0.0 or tilt != 0.0:
                move_camera(pan, tilt)

            centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.drawMarker(annotated, centre, (0, 255, 255),
                           cv2.MARKER_CROSS, 24, 2)

        # ── Info overlay – top right ───────────────────────
        put_text_right(annotated, f"Persons: {person_count}",  row=0, color=(255, 255, 255))
        put_text_right(annotated, f"cx: {cx:.2f}  cy: {cy:.2f}", row=1, color=(0, 255, 255))
        put_text_right(annotated, f"pan: {pan:.2f}  tilt: {tilt:.2f}", row=2, color=(0, 255, 255))

        cv2.imshow("YOLOv8 Person Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection complete.")