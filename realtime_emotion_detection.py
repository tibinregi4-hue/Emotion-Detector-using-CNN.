import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 48 → 24
        x = self.pool(F.relu(self.conv2(x)))  # 24 → 12
        x = self.pool(F.relu(self.conv3(x)))  # 12 → 6

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

py_file_path = os.path.abspath(__file__)
curr_folder = os.path.dirname(py_file_path)
model_path = os.path.join(curr_folder, "emotion_cnn.pth")
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=7).to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()


### Face Detection, Frame Processing and ROI Extraction ###
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam!!!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame!!!")
        break

    # Convert frame to grayscale for detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    # Positive x -> Right
    # Positive y -> Downward
    for (x, y, w, h) in faces:  # min x (left), min y (top), width, height
        x1, y1 = x, y             # top-left
        x2, y2 = x + w, y + h + int(0.15 * h)    # bottom-right (bottom expand 15%)

        # Clip to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # Draw face rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop face ROI
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        # Frame processing
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_48 = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)

        # Scale to 0..1
        x_in = face_48.astype(np.float32) / 255.0
        x_in = (x_in - 0.5) / 0.5 # --> 2*x - 1

        # Make tensor: (1, 1, 48, 48)
        face_tensor = torch.from_numpy(x_in).unsqueeze(0).unsqueeze(0).to(device)

        # Emotion prediction
        with torch.no_grad():
            logits = model(face_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        emotion = EMOTIONS[int(pred.item())]
        confidence = float(conf.item())

        label = f"{emotion} ({confidence:.2f})"

        # Put label above rectangle
        cv2.putText(
            frame, label, (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # just for debugging
        # cv2.imshow("Face ROI", face_roi)
        # cv2.imshow("Face 48x48", face_48)

    # Show webcam frame
    cv2.imshow("Webcam - Emotion Detection", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
