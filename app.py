import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from catboost import CatBoostClassifier
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase

# --- CONFIG ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
RESNET_PATH = 'models/best_resnet50.pth'
CATBOOST_PATH = 'models/fer_catboost.cbm'
IMG_SIZE = 224
LANDMARK_PAIRS = [(61, 291), (0, 17), (13, 14), (78, 308), (159, 145), (386, 374), (33, 133), (362, 263), (65, 159), (295, 386), (70, 63), (300, 293), (1, 152), (1, 0), (234, 454), (172, 397), (5, 195), (4, 1), (19, 2)]

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = models.resnet50(weights=None)
    cnn.fc = nn.Sequential(
        nn.BatchNorm1d(2048), nn.Dropout(0.5), 
        nn.Linear(2048, 512), nn.ReLU(), 
        nn.BatchNorm1d(512), nn.Dropout(0.4), 
        nn.Linear(512, len(CLASSES))
    )
    ckpt = torch.load(RESNET_PATH, map_location=device)
    cnn.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
    cnn.to(device).eval()
    cb = CatBoostClassifier().load_model(CATBOOST_PATH)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    return cnn, cb, face_mesh, device

class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.cnn, self.cb, self.mesh, self.dev = load_models()
        self.mp_drawing = mp.solutions.drawing_utils
        # Update Mesh color to match a subtle grey so the labels stand out
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, color=(100, 100, 100))
        
        self.smooth_cnn = np.zeros(len(CLASSES))
        self.smooth_stacking = np.zeros(len(CLASSES))
        self.smoothing = 0.3

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        results = self.mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            self.mp_drawing.draw_landmarks(img, landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION, None, self.drawing_spec)
            
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            x1, y1 = int(min(coords[:,0]) * w), int(min(coords[:,1]) * h)
            x2, y2 = int(max(coords[:,0]) * w), int(max(coords[:,1]) * h)
            
            crop = img_rgb[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                # 1. ResNet-50 Branch
                pil_crop = Image.fromarray(crop)
                tx = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.Grayscale(3), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    p_rn = torch.softmax(self.cnn(tx(pil_crop).unsqueeze(0).to(self.dev)), dim=1).cpu().numpy()[0]
                
                # 2. CatBoost Branch
                iod = np.linalg.norm(coords[[362, 263]].mean(0)[:2] - coords[[33, 133]].mean(0)[:2]) + 1e-6
                c_norm = (coords - coords.mean(0)) / iod
                feat_cb = np.concatenate([c_norm.flatten(), [np.linalg.norm(c_norm[a] - c_norm[b]) for a, b in LANDMARK_PAIRS]]).reshape(1, -1)
                p_cb = self.cb.predict_proba(feat_cb)[0]
                
                # 3. Ensemble
                p_stacking = 0.55 * p_rn + 0.45 * p_cb
                
                self.smooth_cnn = (self.smooth_cnn * (1 - self.smoothing)) + (p_rn * self.smoothing)
                self.smooth_stacking = (self.smooth_stacking * (1 - self.smoothing)) + (p_stacking * self.smoothing)
                
                idx_cnn = np.argmax(self.smooth_cnn)
                idx_stack = np.argmax(self.smooth_stacking)
                
                label_cnn = f"CNN ONLY: {CLASSES[idx_cnn]} ({int(self.smooth_cnn[idx_cnn]*100)}%)"
                label_stack = f"STACKING: {CLASSES[idx_stack]} ({int(self.smooth_stacking[idx_stack]*100)}%)"
                
                # COLORS
                # Blue: (255, 100, 0)
                # Neon Pink: (255, 0, 255) 
                
                cv2.putText(img, label_cnn, (x1, max(0, y1-35)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
                cv2.putText(img, label_stack, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        return img

# --- APP UI ---
st.title("🔬 Neural Emotion Comparison Engine")
st.markdown("""
    Comparing **ResNet-50 (CNN)** against your **Stacking Ensemble**.
    - <span style='color:#0064FF; font-weight:bold'>BLUE:</span> Standard CNN Result
    - <span style='color:#FF00FF; font-weight:bold'>NEON PINK:</span> Ensemble Stacking (Final)
""", unsafe_allow_html=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="emotion-comparison", 
    video_transformer_factory=EmotionTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    async_processing=True,
)
