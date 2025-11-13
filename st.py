import os, time, sys
import numpy as np
import cv2
import mediapipe as mp
import subprocess

# NEW: Platform-specific import for sound
# This will only import winsound if the script is running on Windows
if sys.platform == "win32":
    import winsound

MODEL_PATH = "drowsiness_mobilenetv2.h5"
IMG_SIZE = 145
USE_MODEL = os.path.exists(MODEL_PATH)

if USE_MODEL:
    try:
        import tensorflow as tf

        # FIX: Define a custom layer wrapper to fix the 'groups' argument error
        # This class intercepts the load-time arguments, removes the
        # unrecognized 'groups' keyword, and passes the rest to the
        # real DepthwiseConv2D layer.
        class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                # Pop (remove) the 'groups' argument if it exists
                kwargs.pop('groups', None)
                # Call the parent's constructor with the cleaned arguments
                super().__init__(*args, **kwargs)

        # FIX: Tell load_model to use our fixed class
        # whenever it sees 'DepthwiseConv2D' in the H5 file
        custom_objects = {'DepthwiseConv2D': FixedDepthwiseConv2D}
        
        # FIX: Pass the custom_objects dictionary to load_model
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        
        MODEL_LABELS = ["yawn", "no_yawn", "Closed", "Open"]
        print("[INFO] CNN Model Loaded Successfully ✅")
    except Exception as e:
        USE_MODEL = False
        print(f"[WARN] Model load failed: {e}. Continuing without CNN.")
else:
    print("[INFO] Model file not found. Continuing without CNN.")

# ==================  MODIFIED: CROSS-PLATFORM BEEP  ==================
def beep():
    if not os.path.exists("alarm.wav"):
        print("\a") # Fallback if alarm.wav is missing
        return
    
    try:
        if sys.platform == "win32":
            # Windows (your "local" machine)
            winsound.PlaySound("alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif sys.platform == "darwin":
            # macOS
            subprocess.Popen(["afplay", "alarm.wav"])
        elif sys.platform.startswith("linux"):
            # Linux
            subprocess.Popen(["aplay", "alarm.wav"])
        else:
            # Other/Unknown OS
            print("\a")
    except Exception as e:
        print(f"[WARN] Failed to play sound on platform '{sys.platform}': {e}")
        print("\a") # Fallback to terminal beep

# ==================  MEDIAPIPE SETUP  ==================
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_H = (61, 291)
MOUTH_V = (13, 14)

def _p2xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def _dist(a,b):
    return float(np.linalg.norm(a-b))

def ear(lm, w, h, idx):
    p = [_p2xy(lm[i],w,h) for i in idx]
    return (_dist(p[1],p[5]) + _dist(p[2],p[4])) / (2*_dist(p[0],p[3]) + 1e-6)

def mar(lm,w,h):
    pL=_p2xy(lm[MOUTH_H[0]],w,h)
    pR=_p2xy(lm[MOUTH_H[1]],w,h)
    pU=_p2xy(lm[MOUTH_V[0]],w,h)
    pD=_p2xy(lm[MOUTH_V[1]],w,h)
    return _dist(pU,pD)/(_dist(pL,pR)+1e-6)

# ==================  THRESHOLDS  ==================
EAR_THRESH = 0.19
MAR_THRESH = 0.45
YAWN_FRAMES = 4
EYE_CLOSED_SECONDS_LIMIT = 3.0       # >3s eyes closed → drowsy
BEEP_COOLDOWN = 5.0                  # seconds

def main():
    cap = cv2.VideoCapture(0)

    eye_closed_start = None
    yawn_counter = 0
    last_beep_time = 0

    fps_t = time.time()
    fps = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = FACE_MESH.process(rgb)

        eye_closed = False
        yawning = False
        drowsy = False
        reasons = []

        # ---------------- FACE NOT DETECTED = DROWSY ----------------
        if not res.multi_face_landmarks:
            drowsy = True
            reasons.append("FACE / EYES NOT VISIBLE")

        else:
            lm = res.multi_face_landmarks[0].landmark
            
            # ---------------- CHOOSE LOGIC (CNN or GEOMETRIC) ----------------
            if USE_MODEL:
                # Get bounding box for face crop from landmarks
                all_x = [l.x * w for l in lm]
                all_y = [l.y * h for l in lm]
                x1, y1 = int(min(all_x)), int(min(all_y))
                x2, y2 = int(max(all_x)), int(max(all_y))

                # Add padding to get a better crop
                pad_x = int((x2 - x1) * 0.15)
                pad_y = int((y2 - y1) * 0.15)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)

                # Crop, Preprocess, and Predict
                face_crop = frame[y1:y2, x1:x2]
                
                pred_label = "Unknown"
                if face_crop.size > 0:
                    try:
                        resized_face = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                        normalized_face = resized_face / 255.0
                        input_data = np.expand_dims(normalized_face, axis=0)
                        
                        prediction = model.predict(input_data, verbose=0) # verbose=0 to hide print
                        pred_label = MODEL_LABELS[np.argmax(prediction[0])]

                        # Set flags based on model prediction
                        eye_closed = (pred_label == "Closed")
                        yawning = (pred_label == "yawn")
                    except Exception as e:
                        print(f"[ERROR] Model prediction failed: {e}")
                        pred_label = "Error"
                
                # Display model prediction and bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame,f"CNN: {pred_label}",(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            else:
                # Fallback to geometric method if model not used
                e = (ear(lm,w,h,LEFT_EYE)+ear(lm,w,h,RIGHT_EYE))/2
                m = mar(lm,w,h)

                eye_closed = e < EAR_THRESH
                yawning = m > MAR_THRESH

                cv2.putText(frame,f"EAR:{e:.3f}",(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                cv2.putText(frame,f"MAR:{m:.3f}",(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            
            # ---------------- (This logic is the same, but uses flags from above) ----------------

            # Yawning detection (must persist)
            if yawning:
                yawn_counter += 1
            else:
                yawn_counter = 0
            if yawn_counter >= YAWN_FRAMES:
                drowsy = True
                reasons.append("YAWNING")

            # Eyes closed time-based detection
            if eye_closed:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start >= EYE_CLOSED_SECONDS_LIMIT:
                    drowsy = True
                    reasons.append(f"EYES CLOSED > {int(EYE_CLOSED_SECONDS_LIMIT)}s")
            else:
                eye_closed_start = None

        # ---------------- DROWSINESS ALERT + COOLDOWN ----------------
        if drowsy:
            if time.time() - last_beep_time >= BEEP_COOLDOWN:
                beep()
                last_beep_time = time.time()

            cv2.rectangle(frame,(0,0),(w,60),(0,0,255),-1)
            cv2.putText(frame,"DROWSY! " + " + ".join(reasons),(10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        # FPS display
        now=time.time()
        if now-fps_t>=0.5:
            fps=1/(now-fps_t)
            fps_t=now
        cv2.putText(frame,f"FPS:{fps:.1f}",(w-120,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.imshow("Drowsiness Detection (q to quit)",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()