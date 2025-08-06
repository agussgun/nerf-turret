import cv2
import mediapipe as mp
import time
import numpy as np
import serial

# 🔌 Serial Arduino - sesuaikan port & baudrate
arduino = serial.Serial("COM3", 9600, timeout=1)  # Ganti 'COM3' sesuai port kamu
time.sleep(2)  # Tunggu koneksi serial

# Konfigurasi MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Thresholds
EYE_AR_THRESH = 0.25
SLEEP_THRESHOLD_SEC = 10  # Durasi untuk anggap tertidur

# Landmark mata dari MediaPipe FaceMesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def eye_aspect_ratio(eye_points, landmarks):
    p1 = np.array(landmarks[eye_points[1]])
    p2 = np.array(landmarks[eye_points[5]])
    p3 = np.array(landmarks[eye_points[2]])
    p4 = np.array(landmarks[eye_points[4]])
    p5 = np.array(landmarks[eye_points[0]])
    p6 = np.array(landmarks[eye_points[3]])
    ear = (euclidean_distance(p1, p2) + euclidean_distance(p3, p4)) / (
        2.0 * euclidean_distance(p5, p6)
    )
    return ear


# 📷 IP Kamera atau Webcam
cap = cv2.VideoCapture("http://192.168.1.67:3001/video")  # Ganti sesuai IP cam kamu

prev_time = 0
eye_closed_time = None
asleep = False  # Status apakah sudah kirim sinyal tidur

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("❌ Gagal membaca frame dari IP Cam.")
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        frame_h, frame_w = frame.shape[:2]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [
                    (int(lm.x * frame_w), int(lm.y * frame_h))
                    for lm in face_landmarks.landmark
                ]

                left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
                right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
                avg_ear = (left_ear + right_ear) / 2.0

                mata_tertutup = avg_ear < EYE_AR_THRESH

                if mata_tertutup:
                    if eye_closed_time is None:
                        eye_closed_time = time.time()
                    elapsed = time.time() - eye_closed_time

                    if elapsed >= SLEEP_THRESHOLD_SEC and not asleep:
                        arduino.write(b"1")
                        print("📩 Kirim 1 (Tertidur)")
                        asleep = True
                        cv2.putText(
                            frame,
                            "Tertidur",
                            (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            3,
                        )
                else:
                    if asleep:
                        arduino.write(b"0")
                        print("📩 Kirim 0 (Bangun)")
                        asleep = False
                    eye_closed_time = None

                # Gambar status & landmark
                status = "Tertutup" if mata_tertutup else "Terbuka"
                color = (0, 0, 255) if mata_tertutup else (0, 255, 0)

                cv2.putText(
                    frame,
                    f"Mata: {status}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )
                for idx in LEFT_EYE + RIGHT_EYE:
                    cv2.circle(frame, landmarks[idx], 2, color, -1)

        # Tampilkan FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (100, 255, 100),
            2,
        )

        resized_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Deteksi Mata via IP Cam", resized_frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
