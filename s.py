import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import numpy as np
import cv2
import mediapipe as mp

# Optional tensorflow import for model-based drowsiness
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_PATH = "drowsiness_mobilenetv2.h5"
IMG_SIZE = 145
USE_MODEL = False
MODEL_LABELS = ["yawn", "no_yawn", "Closed", "Open"]

if TF_AVAILABLE:
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}
        )
        USE_MODEL = True
    except Exception as e:
        st.warning(f"Model load failed: {e}. Using geometric fallback.")

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_H = (61, 291)
MOUTH_V = (13, 14)
EAR_THRESH = 0.19
MAR_THRESH = 0.45
YAWN_FRAMES = 4
EYE_CLOSED_SECONDS_LIMIT = 3.0

def _p2xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def _dist(a, b):
    return float(np.linalg.norm(a - b))

def ear(lm, w, h, idx):
    p = [_p2xy(lm[i], w, h) for i in idx]
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2 * _dist(p[0], p[3]) + 1e-6)

def mar(lm, w, h):
    pL = _p2xy(lm[MOUTH_H[0]], w, h)
    pR = _p2xy(lm[MOUTH_H[1]], w, h)
    pU = _p2xy(lm[MOUTH_V[0]], w, h)
    pD = _p2xy(lm[MOUTH_V[1]], w, h)
    return _dist(pU, pD) / (_dist(pL, pR) + 1e-6)

import av  # required for streamlit-webrtc

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.eye_closed_start = None
        self.yawn_counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        eye_closed = False
        yawning = False
        drowsy = False

        if not res.multi_face_landmarks:
            drowsy = True
            status = "FACE / EYES NOT VISIBLE"
        else:
            lm = res.multi_face_landmarks[0].landmark
            if USE_MODEL:
                all_x = [l.x * w for l in lm]
                all_y = [l.y * h for l in lm]
                x1, y1 = int(min(all_x)), int(min(all_y))
                x2, y2 = int(max(all_x)), int(max(all_y))
                pad_x = int((x2 - x1) * 0.15)
                pad_y = int((y2 - y1) * 0.15)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                face_crop = img[y1:y2, x1:x2]
                pred_label = "Unknown"
                if face_crop.size > 0:
                    try:
                        resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                        normalized = resized / 255.0
                        input_data = np.expand_dims(normalized, axis=0)
                        pred = model.predict(input_data, verbose=0)
                        pred_label = MODEL_LABELS[np.argmax(pred[0])]
                        eye_closed = (pred_label == "Closed")
                        yawning = (pred_label == "yawn")
                    except Exception:
                        pred_label = "Error"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(img, f"CNN: {pred_label}", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
            else:
                e = (ear(lm, w, h, LEFT_EYE) + ear(lm, w, h, RIGHT_EYE)) / 2
                m = mar(lm, w, h)
                eye_closed = e < EAR_THRESH
                yawning = m > MAR_THRESH
                cv2.putText(img, f"EAR:{e:.3f}", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(img, f"MAR:{m:.3f}", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if yawning:
                self.yawn_counter += 1
            else:
                self.yawn_counter = 0

            if self.yawn_counter >= YAWN_FRAMES:
                drowsy = True
                status = "YAWNING"

            if eye_closed:
                if self.eye_closed_start is None:
                    self.eye_closed_start = cv2.getTickCount() / cv2.getTickFrequency()
                elif (cv2.getTickCount() / cv2.getTickFrequency() - self.eye_closed_start) >= EYE_CLOSED_SECONDS_LIMIT:
                    drowsy = True
                    status = f"EYES CLOSED > {int(EYE_CLOSED_SECONDS_LIMIT)}s"
            else:
                self.eye_closed_start = None

        if drowsy:
            cv2.rectangle(img, (0, 0), (w, 60), (0,0,255), -1)
            cv2.putText(img, "DROWSY! "+status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Webcam Drowsiness Detection App")
st.markdown("Detects drowsiness (yawn/eyes closed) in real time via webcam")

webrtc_streamer(key="drowsy-detect", video_processor_factory=DrowsinessProcessor)
