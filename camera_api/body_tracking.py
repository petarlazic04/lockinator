import cv2
import numpy as np
from ultralytics import YOLO
import torch
from camera import Camera
from deepface import DeepFace
import threading
from queue import Queue, Empty
from pathlib import Path
import time
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime

# ── Configuration ─────────────────────────────────────────
CAM_IP         = "192.168.50.222"
CAM_PORT       = 8899
CAM_USER       = "admin"
CAM_PASS       = "admin123"
RTSP_URL       = f"rtsp://{CAM_IP}:554/live/ch00_0"
MODEL_PATH     = "yolov8n-face.pt"
DB_PATH        = Path(__file__).resolve().parent / "pictures"

FRAME_W, FRAME_H = 960, 540
DEAD_ZONE      = 0.25
PAN_SPEED      = 0.2
TILT_SPEED     = 0.2
MOVE_DURATION  = 0.01
CONF_THRESHOLD = 0.5
RECOGNITION_EVERY = 45
ENABLE_FACE_RECOGNITION = True
FACE_SIMILARITY_THRESHOLD = 0.60
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
# ──────────────────────────────────────────────────────────

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
RECORD_FPS = 20
RECORD_BUFFER_SECONDS = 5.0
RECORD_CODEC = "mp4v"

NTFY_TOPIC = "lockinator-alarm-changeme"

def send_ntfy_notification(title: str, message: str, frame=None):
    def _send():
        try:
            if frame is not None:
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    req = urllib.request.Request(
                        f"https://ntfy.sh/{NTFY_TOPIC}",
                        data=buf.tobytes(),
                        headers={
                            "Title": title,
                            "Message": message,
                            "Priority": "urgent",
                            "Tags": "rotating_light,camera",
                            "Content-Type": "image/jpeg",
                            "Filename": "intruder.jpg",
                        },
                    )
                    urllib.request.urlopen(req, timeout=10)
                    return
            req = urllib.request.Request(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=message.encode(),
                headers={
                    "Title": title,
                    "Priority": "urgent",
                    "Tags": "rotating_light,camera",
                },
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[WARN] ntfy notification failed: {e}")
    threading.Thread(target=_send, daemon=True).start()


# Alarm sound configuration
ALARM_MP3_PATH = Path(__file__).resolve().parent / "alarm.mp3"
_alarm_stop_event = threading.Event()
_alarm_thread = None


def _alarm_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(ALARM_MP3_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            while proc.poll() is None:
                if stop_event.is_set():
                    proc.terminate()
                    return
                time.sleep(0.05)
        except Exception as e:
            print(f"[WARN] Alarm playback error: {e}")
            return


def start_alarm():
    global _alarm_thread, _alarm_stop_event
    if _alarm_thread is not None and _alarm_thread.is_alive():
        return
    _alarm_stop_event.clear()
    _alarm_thread = threading.Thread(target=_alarm_loop, args=(_alarm_stop_event,), daemon=True)
    _alarm_thread.start()
    print("[INFO] Alarm started")


def stop_alarm():
    global _alarm_thread
    _alarm_stop_event.set()
    if _alarm_thread is not None:
        _alarm_thread.join(timeout=1.0)
        _alarm_thread = None
    print("[INFO] Alarm stopped")


cam   = Camera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(RTSP_URL)

DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"[INFO] YOLO inference device: {DEVICE_NAME}")

_last_log_time = defaultdict(lambda: 0.0)
def log_throttle(key: str, msg: str, interval: float = 5.0):
    now = time.time()
    if now - _last_log_time[key] >= interval:
        print(msg)
        _last_log_time[key] = now

cv2.namedWindow("YOLOv8 Person Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Person Detection", FRAME_W, FRAME_H)

cam_busy = False
reference_embeddings = []
reference_embeddings_ready = False
reference_embeddings_loading = False
reference_embeddings_lock = threading.Lock()
recognition_queue = Queue(maxsize=1)
recognition_result_lock = threading.Lock()
latest_face_label = "Searching..."
latest_face_distance = None
HOST_NAMES = {"pero", "djuro"}
HOST_TOGGLE_COOLDOWN = 3.0
host_present = False
host_next_toggle_at = 0.0
latest_recognized_box = None
host_confirm_start = None
host_confirmed = False
recording = False
record_writer = None
recording_stop_at = 0.0
saved_pan = None
saved_tilt = None

HARD_PAN = 0.0
HARD_TILT = 0.0
KNOWN_NAMES = set()
INTRUDER_TIMEOUT = 10.0
intruder_present = False
intruder_expires = 0.0
intruder_box = None
MAINTAIN_HOME_INTERVAL = 5.0
maintain_home_thread_started = False


def iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea) + 1e-10
    return interArea / denom


def cosine_similarity(vec1, vec2):
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-10
    if denom == 0:
        return -1.0
    return float(np.dot(vec1, vec2) / denom)


def load_reference_embeddings(db_path: Path):
    if not db_path.exists():
        print(f"[WARN] DB folder does not exist: {db_path}")
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    embeddings = []
    for image_path in db_path.rglob("*"):
        if image_path.suffix.lower() not in image_extensions:
            continue

        person_name = image_path.parent.name or image_path.stem
        try:
            representation = DeepFace.represent(
                img_path=str(image_path),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
            )
        except Exception as exc:
            print(f"[WARN] Cannot load reference {image_path}: {exc}")
            continue

        if not representation:
            continue

        if isinstance(representation, list):
            representation = representation[0]

        embedding = representation.get("embedding") if isinstance(representation, dict) else None
        if embedding is None:
            continue

        embeddings.append((person_name, np.asarray(embedding, dtype=np.float32)))

    print(f"[INFO] Loaded {len(embeddings)} reference embeddings")
    return embeddings


def load_reference_embeddings_async(db_path: Path):
    global reference_embeddings, reference_embeddings_ready, reference_embeddings_loading

    if reference_embeddings_loading or reference_embeddings_ready:
        return

    reference_embeddings_loading = True

    def _worker():
        global reference_embeddings, reference_embeddings_ready, reference_embeddings_loading
        try:
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            image_paths = []

            if db_path.exists():
                image_paths = [
                    image_path
                    for image_path in db_path.rglob("*")
                    if image_path.suffix.lower() in image_extensions
                ]
            else:
                print(f"[WARN] DB folder does not exist: {db_path}")

            with reference_embeddings_lock:
                reference_embeddings.clear()

            for image_path in image_paths:
                person_name = image_path.parent.name or image_path.stem
                try:
                    representation = DeepFace.represent(
                        img_path=str(image_path),
                        model_name=MODEL_NAME,
                        detector_backend=DETECTOR_BACKEND,
                        enforce_detection=False,
                    )
                except Exception as exc:
                    print(f"[WARN] Cannot load reference {image_path}: {exc}")
                    continue

                if not representation:
                    continue

                if isinstance(representation, list):
                    representation = representation[0]

                embedding = representation.get("embedding") if isinstance(representation, dict) else None
                if embedding is None:
                    continue

                with reference_embeddings_lock:
                    reference_embeddings.append((person_name, np.asarray(embedding, dtype=np.float32)))
                    KNOWN_NAMES.add(person_name)

            reference_embeddings_ready = True
        finally:
            reference_embeddings_loading = False

    threading.Thread(target=_worker, daemon=True).start()


def recognition_worker():
    global latest_face_label, latest_face_distance
    global host_present, host_next_toggle_at, latest_recognized_box
    global KNOWN_NAMES, intruder_present, intruder_expires, intruder_box
    global host_confirm_start, host_confirmed, maintain_home_thread_started

    while True:
        item = recognition_queue.get()
        if item is None:
            break

        frame, embeddings_snapshot, box = item
        label, distance = recognize_face(frame, embeddings_snapshot)

        with recognition_result_lock:
            latest_face_label = label
            latest_face_distance = distance

        # Toggle host mode only on host detections, with cooldown to avoid double-triggering from one pass.
        if label in HOST_NAMES:
            now = time.time()
            if now < host_next_toggle_at:
                continue

            host_next_toggle_at = now + HOST_TOGGLE_COOLDOWN
            latest_recognized_box = box

            if not host_present:
                host_present = True
                # clear intruder state
                intruder_present = False
                intruder_box = None
                host_confirmed = True
                # move to host home view
                threading.Thread(target=_goto_host_view, daemon=True).start()
                # start maintain thread if not started
                if not maintain_home_thread_started:
                    maintain_home_thread_started = True
                    threading.Thread(target=_maintain_home_view, daemon=True).start()
            else:
                host_present = False
                host_confirmed = False
                latest_recognized_box = None
                threading.Thread(target=_restore_prev_view, daemon=True).start()
        else:
            # known person (guest) — update latest_recognized_box
            if label in KNOWN_NAMES:
                latest_recognized_box = box
            else:
                # Unknown person / not in known group
                # Only mark as intruder after embeddings are fully loaded
                if not host_present and reference_embeddings_ready:
                    intruder_present = True
                    intruder_expires = time.time() + INTRUDER_TIMEOUT
                    intruder_box = box


def recognize_face(frame, reference_embeddings):
    try:
        representation = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
    except Exception as exc:
        log_throttle('deepface_error', f"[WARN] DeepFace error: {exc}", interval=10.0)
        return "Unknown", None

    if not representation:
        return "Unknown", None

    if isinstance(representation, list):
        representation = representation[0]

    embedding = representation.get("embedding") if isinstance(representation, dict) else None
    if embedding is None or not reference_embeddings:
        return "Unknown", None

    best_person = "Unknown"
    best_score = -1.0
    embedding = np.asarray(embedding, dtype=np.float32)

    for person_name, reference_embedding in reference_embeddings:
        score = cosine_similarity(embedding, reference_embedding)
        if score > best_score:
            best_score = score
            best_person = person_name

    if best_score < FACE_SIMILARITY_THRESHOLD:
        return "Unknown", None

    return best_person, best_score


def crop_for_recognition(frame, box, padding=0.25):
    if box is None:
        return None

    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    h, w = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(w, x2 + pad_x)
    bottom = min(h, y2 + pad_y)

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (320, 320))


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


def _goto_host_view():
    global saved_pan, saved_tilt
    # try to read current absolute position from camera status
    try:
        resp = cam.get_status()
        if resp is not None and resp.content:
            try:
                root = ET.fromstring(resp.content)
                for elem in root.iter():
                    if 'PanTilt' in elem.tag:
                        x = elem.attrib.get('x')
                        y = elem.attrib.get('y')
                        if x is not None and y is not None:
                            try:
                                saved_pan = float(x)
                                saved_tilt = float(y)
                                log_throttle('saved_position', f"[INFO] Saved camera position pan={saved_pan}, tilt={saved_tilt}", interval=30.0)
                            except Exception:
                                pass
                        break
            except Exception as e:
                log_throttle('parse_get_status_failed', f"[WARN] Failed to parse get_status response: {e}", interval=30.0)
    except Exception as e:
        log_throttle('get_status_failed', f"[WARN] get_status failed: {e}", interval=30.0)

    tries = 3
    for attempt in range(tries):
        try:
            resp = cam.absolute_move(pan=HARD_PAN, tilt=HARD_TILT)
            log_throttle('moved_to_home', f"[INFO] Moved to host home view (attempt {attempt+1})", interval=30.0)
            break
        except Exception as e:
            log_throttle('move_to_host_failed', f"[WARN] Failed to move to host view (attempt {attempt+1}): {e}", interval=30.0)
            time.sleep(2)


def _restore_prev_view():
    global saved_pan, saved_tilt
    if saved_pan is None or saved_tilt is None:
        log_throttle('no_saved_position', "[INFO] No saved camera position to restore.", interval=30.0)
        return
    tries = 3
    for attempt in range(tries):
        try:
            resp = cam.absolute_move(pan=saved_pan, tilt=saved_tilt)
            log_throttle('restored_camera', f"[INFO] Restored camera to previous position (attempt {attempt+1})", interval=30.0)
            saved_pan = None
            saved_tilt = None
            break
        except Exception as e:
            log_throttle('restore_failed', f"[WARN] Failed to restore camera position (attempt {attempt+1}): {e}", interval=30.0)
            time.sleep(2)


def _maintain_home_view():
    global host_confirmed, maintain_home_thread_started
    while host_confirmed:
        for attempt in range(3):
            try:
                cam.absolute_move(pan=HARD_PAN, tilt=HARD_TILT)
                break
            except Exception as e:
                log_throttle('maintain_home_failed', f"[WARN] maintain_home absolute_move attempt {attempt+1} failed: {e}", interval=30.0)
                time.sleep(1)
        time.sleep(MAINTAIN_HOME_INTERVAL)
    maintain_home_thread_started = False


def _ensure_recordings_dir():
    try:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def start_recording(frame):
    global recording, record_writer, recording_stop_at
    if recording:
        return
    _ensure_recordings_dir()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RECORDINGS_DIR / f"intruder_{now}.mp4"
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*RECORD_CODEC)
    try:
        record_writer = cv2.VideoWriter(str(filename), fourcc, RECORD_FPS, (w, h))
        recording = True
        recording_stop_at = 0.0
        print(f"[INFO] Started recording: {filename}")
        start_alarm()
        local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        send_ntfy_notification("ALARM - Lockinator", f"Intruder detected! {local_time}", frame=frame)
    except Exception as e:
        print(f"[WARN] Cannot start recording: {e}")


def stop_recording():
    global recording, record_writer, recording_stop_at
    if not recording:
        return
    try:
        if record_writer is not None:
            record_writer.release()
    except Exception:
        pass
    recording = False
    record_writer = None
    recording_stop_at = 0.0
    stop_alarm()
    print("[INFO] Recording stopped")


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
    print(f"[INFO] DB: {DB_PATH}")
    print("[INFO] Loading face recognition reference embeddings in background...")
    load_reference_embeddings_async(DB_PATH)
    threading.Thread(target=recognition_worker, daemon=True).start()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            log_throttle('frame_read_failed', "[WARN] Frame read failed, retrying...", interval=5.0)
            time.sleep(0.1)
            continue

        frame_count += 1

        h, w = frame.shape[:2]
        results = list(model(frame, stream=True, conf=CONF_THRESHOLD, device=DEVICE))

        annotated = frame.copy()
        person_count = 0
        with recognition_result_lock:
            face_label = latest_face_label if ENABLE_FACE_RECOGNITION else "Disabled"
            face_distance = latest_face_distance if ENABLE_FACE_RECOGNITION else None

        # Build list of detection boxes
        detection_list = []
        for result in results:
            for detection_box in result.boxes:
                cls  = int(detection_box.cls[0])
                conf = float(detection_box.conf[0])
                if cls != 0 or conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, detection_box.xyxy[0])
                detection_list.append(((x1, y1, x2, y2), conf))

        # choose box to recognize every Nth frame (largest detected person)
        recognition_box = None
        if detection_list and frame_count % RECOGNITION_EVERY == 0:
            recognition_box = max(detection_list, key=lambda b: (b[0][2]-b[0][0])*(b[0][3]-b[0][1]))[0]
            recognition_frame = crop_for_recognition(frame, recognition_box)
            if recognition_frame is not None:
                with reference_embeddings_lock:
                    current_embeddings = list(reference_embeddings)

                if current_embeddings:
                    try:
                        recognition_queue.put_nowait((recognition_frame, current_embeddings, recognition_box))
                    except Exception:
                        pass

        # expire intruder flag if timeout passed
        if intruder_present and time.time() > intruder_expires:
            intruder_present = False
            intruder_box = None

        for box, conf in detection_list:
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            person_count += 1

            # always draw detections, but do not mark intruders when host is present
            draw_color = (255, 0, 0)
            thickness = 2
            label_txt = f"Person {person_count} {conf:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), draw_color, thickness)
            cv2.putText(annotated, label_txt, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)

        cx, cy, pan, tilt = 0.0, 0.0, 0.0, 0.0

        # Decide which box to track: do not track anyone while host is present
        track_box = None
        if not host_present and detection_list:
            track_box = max(detection_list, key=lambda b: (b[0][2]-b[0][0])*(b[0][3]-b[0][1]))[0]

        if host_confirmed:
            # show overlay that host is present and camera is on home view
            put_text_right(annotated, "Host present - home view", row=5, color=(0, 0, 0))
        # show recording overlay if active
        if recording:
            put_text_right(annotated, "REC: recording...", row=6, color=(0, 0, 255))
        # start/stop recording logic: when an intruder (unknown person) is present start recording
        if intruder_present and not recording:
            start_recording(frame)
        if recording:
            # write frame to file
            try:
                if record_writer is not None:
                    record_writer.write(frame)
            except Exception:
                pass
            # if intruder disappeared, schedule stop after buffer
            if not intruder_present:
                if recording_stop_at == 0.0:
                    recording_stop_at = time.time() + RECORD_BUFFER_SECONDS
                elif time.time() >= recording_stop_at:
                    stop_recording()
        if track_box is not None:
            x1, y1, x2, y2 = track_box
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
        put_text_right(annotated, f"Face: {face_label}", row=1, color=(0, 0, 0) if face_label != "Unknown" else (0, 0, 255))
        put_text_right(annotated, f"cx: {cx:.2f}  cy: {cy:.2f}", row=2, color=(0, 255, 255))
        put_text_right(annotated, f"pan: {pan:.2f}  tilt: {tilt:.2f}", row=3, color=(0, 255, 255))
        if face_distance is not None:
            put_text_right(annotated, f"dist: {face_distance:.4f}", row=4, color=(255, 255, 0))

        cv2.imshow("YOLOv8 Person Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection complete.")