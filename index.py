import sys
import cv2
import os
import numpy as np
import torch
from deepface import DeepFace
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QPoint, QPropertyAnimation, Qt
import smtplib
from playsound import playsound

# Directory where the known faces dataset is stored
dataset_dir = r'C:\Users\Vijay\Downloads\int 3'

# Threat categories
threat_categories = ['knife', 'gun', 'rifle']  # Modify as needed


# Load known faces from the dataset directory
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.png')]
    total_files = len(files)
    for i, filename in enumerate(files):
        image_path = os.path.join(dataset_dir, filename)
        encoding = DeepFace.represent(img_path=image_path, model_name="Facenet")[0]["embedding"]
        known_face_encodings.append(np.array(encoding))
        known_face_names.append(os.path.splitext(filename)[0])
        print(f"Loaded {i + 1}/{total_files}: {filename}")
    return known_face_encodings, known_face_names


# Alert system: sound alert
def sound_alert():
    playsound('alert_sound.mp3')


# Alert system: email alert
def email_alert(threat_label):
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login('your_email@gmail.com', 'your_password')
        message = f"Subject: Threat Alert!\n\nA {threat_label} has been detected!"
        server.sendmail('your_email@gmail.com', 'recipient_email@gmail.com', message)
        server.quit()
        print("Email alert sent!")
    except Exception as e:
        print(f"Failed to send email alert: {e}")


known_face_encodings, known_face_names = load_known_faces()

# Load pre-trained object detection model (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def detect_objects(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img)
    return results


def recognize_faces(frame, known_face_encodings, known_face_names):
    try:
        faces = DeepFace.find(img_path=frame, db_path=dataset_dir, model_name="Facenet")
        face_locations = []
        face_names = []
        for face in faces:
            face_locations.append(face['region'])
            face_names.append(face['identity'])
    except:
        face_locations = []
        face_names = []
    return face_locations, face_names


class ThreatDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = cv2.VideoCapture(0)

        # Set camera properties for smoother frame rate
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.threat_detected = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS (1000 ms / 30 FPS = 33 ms per frame)

    def initUI(self):
        self.setWindowTitle('Threat Detection System')
        self.setGeometry(100, 100, 800, 600)

        # Set background gradient
        self.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1c1c2d, stop:1 #282846);")

        self.image_label = QLabel(self)
        self.image_label.setGeometry(20, 20, 640, 480)
        self.image_label.setStyleSheet("border: 2px solid #3d3d5c;")

        self.status_label = QLabel(self)
        self.status_label.setText('Status: Monitoring...')
        self.status_label.setGeometry(20, 520, 600, 40)
        self.status_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")

        self.quit_button = QPushButton('Quit', self)
        self.quit_button.setGeometry(680, 520, 100, 40)
        self.quit_button.setStyleSheet(
            "background-color: #d9534f; color: white; font-size: 16px; font-weight: bold; border-radius: 10px;")
        self.quit_button.clicked.connect(self.close_application)

        self.show()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Detect objects in the frame
        results = detect_objects(frame)

        # Recognize faces in the frame
        face_locations, face_names = recognize_faces(frame, known_face_encodings, known_face_names)

        # Draw bounding boxes for detected objects
        for *xyxy, conf, cls in results.xyxy[0]:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Check for threats
            if model.names[int(cls)] in threat_categories:
                if not self.threat_detected:
                    print(f"Threat detected: {label}")
                    self.status_label.setText(f'Status: Threat detected ({label})!')
                    self.animate_status_label()
                    sound_alert()
                    email_alert(model.names[int(cls)])
                    self.threat_detected = True

        # Draw bounding boxes for recognized faces
        for (x, y, w, h), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert the frame to QImage for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        step = channel * width
        qimg = QImage(frame_rgb.data, width, height, step, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

        # Reset threat detection for the next frame
        self.threat_detected = False

    def animate_status_label(self):
        self.animation = QPropertyAnimation(self.status_label, b"pos")
        self.animation.setDuration(500)
        self.animation.setStartValue(self.status_label.pos())
        self.animation.setEndValue(self.status_label.pos() + QPoint(0, -10))
        self.animation.setEasingCurve(Qt.BounceCurve)
        self.animation.start()

    def close_application(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThreatDetectionGUI()
    sys.exit(app.exec_())
