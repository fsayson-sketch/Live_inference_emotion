import os
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
from streamlit_webrtc import (
    webrtc_streamer,
    RTCConfiguration,
    VideoProcessorBase,
)

# --- CONFIG ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

RESNET_PATH = "models/best_resnet50.pth"
CATBOOST_PATH = "models/fer_catboost.cbm"

LANDMARK_PAIRS = [
    (61, 291), (0, 17), (13, 14), (78, 308),
    (159, 145), (386, 374), (33, 133), (362, 263),
    (65, 159), (295, 386), (70, 63), (300, 293),
    (1, 152), (1, 0), (234, 454), (172, 397),
    (5, 195), (4, 1), (19, 2)
]


# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # CNN MODEL
    cnn = models.resnet50(weights=None)

    cnn.fc = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(0.5),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, len(CLASSES)),
    )

    ckpt = torch.load(
        RESNET_PATH,
        map_location=device
    )

    state_dict = (
        ckpt["model_state"]
        if "model_state" in ckpt
        else ckpt
    )

    cnn.load_state_dict(state_dict)
    cnn.to(device)
    cnn.eval()

    # CATBOOST
    cb = CatBoostClassifier()
    cb.load_model(CATBOOST_PATH)

    # NEW MEDIAPIPE TASK API
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(
        model_asset_path="face_landmarker.task"
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    face_landmarker = vision.FaceLandmarker.create_from_options(
        options
    )

    return cnn, cb, face_landmarker, device


# --- VIDEO PROCESSOR ---
class ComparisonProcessor(VideoProcessorBase):
    def __init__(self):
        self.cnn, self.cb, self.mesh, self.dev = load_models()

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1,
            color=(100, 100, 100),
        )

        self.smooth_cnn = np.zeros(len(CLASSES))
        self.smooth_stack = np.zeros(len(CLASSES))

        self.frame_count = 0
        self.cnn_res = "..."
        self.stack_res = "..."

    def recv(self, frame):
    self.frame_count += 1

    img = frame.to_ndarray(format="bgr24")
    h_orig, w_orig, _ = img.shape

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # Lower resolution for cloud CPU performance
    img_small = cv2.resize(
        img_rgb,
        (480, 360)
    )

    # MediaPipe Tasks API
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_small
    )

    results = self.mesh.detect(mp_image)

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]

        # Convert landmarks to numpy
        coords = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ])

        # Predict every 6 frames
        if self.frame_count % 6 == 0:
            x1 = int(
                min(coords[:, 0]) * w_orig
            )

            y1 = int(
                min(coords[:, 1]) * h_orig
            )

            x2 = int(
                max(coords[:, 0]) * w_orig
            )

            y2 = int(
                max(coords[:, 1]) * h_orig
            )

            crop = img_rgb[
                max(0, y1):y2,
                max(0, x1):x2
            ]

            if crop.size > 0:
                pil_crop = Image.fromarray(crop)

                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.Grayscale(3),
                    T.ToTensor(),
                    T.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    ),
                ])

                with torch.no_grad():
                    p_rn = torch.softmax(
                        self.cnn(
                            transform(pil_crop)
                            .unsqueeze(0)
                            .to(self.dev)
                        ),
                        dim=1
                    ).cpu().numpy()[0]

                iod = np.linalg.norm(
                    coords[[362, 263]].mean(0)[:2]
                    - coords[[33, 133]].mean(0)[:2]
                ) + 1e-6

                c_norm = (
                    coords - coords.mean(0)
                ) / iod

                feat_cb = np.concatenate([
                    c_norm.flatten(),
                    [
                        np.linalg.norm(
                            c_norm[a] - c_norm[b]
                        )
                        for a, b in LANDMARK_PAIRS
                    ]
                ]).reshape(1, -1)

                p_cb = self.cb.predict_proba(
                    feat_cb
                )[0]

                p_stack = (
                    0.55 * p_rn
                    + 0.45 * p_cb
                )

                self.smooth_cnn = (
                    self.smooth_cnn * 0.6
                    + p_rn * 0.4
                )

                self.smooth_stack = (
                    self.smooth_stack * 0.6
                    + p_stack * 0.4
                )

                self.cnn_res = (
                    f"CNN: "
                    f"{CLASSES[np.argmax(self.smooth_cnn)]} "
                    f"({int(np.max(self.smooth_cnn) * 100)}%)"
                )

                self.stack_res = (
                    f"STACKING: "
                    f"{CLASSES[np.argmax(self.smooth_stack)]} "
                    f"({int(np.max(self.smooth_stack) * 100)}%)"
                )

        cv2.putText(
            img,
            self.cnn_res,
            (20, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 100, 0),
            2,
        )

        cv2.putText(
            img,
            self.stack_res,
            (20, 80),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 0, 255),
            2,
        )

    return frame.from_ndarray(
        img,
        format="bgr24",
    )


# --- UI ---
st.title("🔬 Cloud Emotion Comparison")

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302"
            ]
        }
    ]
})

webrtc_streamer(
    key="cloud-comparison",
    video_processor_factory=ComparisonProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)