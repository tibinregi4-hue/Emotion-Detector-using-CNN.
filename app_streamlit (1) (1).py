import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# defining model
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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# setup
@st.cache_resource
def load_emotion_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load("emotion_cnn.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_emotion_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# streamlit user interface
st.title("😊 Emotion Detector AI")
st.write("Click 'Start' to open your webcam and detect emotions in real-time.")

run = st.checkbox('Start Webcam')
frame_placeholder = st.empty() # This acts as our "HTML" video container

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access webcam.")
        break

    # processing logic
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        x1, y1, x2, y2 = x, y, x + w, y + h + int(0.15 * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size > 0:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_48 = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
            
            x_in = (face_48.astype(np.float32) / 255.0 - 0.5) / 0.5
            face_tensor = torch.from_numpy(x_in).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(face_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)

            label = f"{EMOTIONS[pred.item()]} ({conf.item():.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # frontend rendering
    # Convert BGR (OpenCV) to RGB (Streamlit)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

cap.release()