import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from collections import deque, Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CONFIDENCE_THRESHOLD = 0.80
SMOOTH_WINDOW        = 10
MIN_VOTES            = 5
WRIST      = 0
FINGERTIPS = [4, 8, 12, 16, 20]
MCP_JOINTS = [1, 5,  9, 13, 17]
PIP_JOINTS = [2, 6, 10, 14, 18]

def extract_normalized_landmarks(hand_landmarks):
    raw   = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    wrist = raw[0]
    centered = [[p[0]-wrist[0], p[1]-wrist[1], p[2]-wrist[2]] for p in raw]
    all_x = [p[0] for p in centered]
    all_y = [p[1] for p in centered]
    scale = max(max(all_x)-min(all_x), max(all_y)-min(all_y))
    if scale == 0: return None
    normed = [[p[0]/scale, p[1]/scale, p[2]/scale] for p in centered]
    return [v for pt in normed for v in pt]

def engineer_features(flat63):
    pts   = np.array(flat63).reshape(21, 3)
    extra = []
    wrist = pts[WRIST]
    for tip in FINGERTIPS:
        extra.append(float(np.linalg.norm(pts[tip] - wrist)))
    for i in range(len(FINGERTIPS)):
        for j in range(i+1, len(FINGERTIPS)):
            extra.append(float(np.linalg.norm(pts[FINGERTIPS[i]] - pts[FINGERTIPS[j]])))
    for k in range(5):
        v1 = pts[PIP_JOINTS[k]] - pts[MCP_JOINTS[k]]
        v2 = pts[FINGERTIPS[k]] - pts[PIP_JOINTS[k]]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        extra.append(float(np.clip(np.dot(v1,v2)/(n1*n2),-1,1)) if n1>0 and n2>0 else 0.0)
    all_x, all_y = pts[:,0], pts[:,1]
    w = max(all_x)-min(all_x)
    h = max(all_y)-min(all_y)
    extra.append(w/h if h > 0 else 1.0)
    return flat63 + extra  # 84 values

model = joblib.load("sign_model.pkl")
le    = joblib.load("label_encoder.pkl")

base_cols  = [f"{ax}{i}" for i in range(21) for ax in ["x","y","z"]]
extra_cols = (
    [f"tip_wrist_{i}" for i in range(5)] +
    [f"tip_dist_{i}_{j}" for i in range(5) for j in range(i+1,5)] +
    [f"curl_{i}" for i in range(5)] +
    ["palm_ratio"]
)
columns = base_cols + extra_cols  # 84

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       model_complexity=1, min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)

prediction_buffer = deque(maxlen=SMOOTH_WINDOW)
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break

    frame   = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    text    = "No hand detected"
    color   = (180, 180, 180)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        flat63 = extract_normalized_landmarks(hand)
        if flat63 is not None:
            feat84     = engineer_features(flat63)
            features   = pd.DataFrame([feat84], columns=columns)
            prediction = model.predict(features)[0]
            confidence = max(model.predict_proba(features)[0])
            label      = le.inverse_transform([prediction])[0]
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_buffer.append(label)
            if len(prediction_buffer) >= MIN_VOTES:
                smoothed = Counter(prediction_buffer).most_common(1)[0][0]
                text  = f"{smoothed}  ({confidence*100:.0f}%)"
                color = (0, 220, 0)
            elif confidence < CONFIDENCE_THRESHOLD:
                text  = f"Low confidence ({confidence*100:.0f}%)"
                color = (0, 165, 255)
            else:
                text  = "Warming up..."
                color = (255, 220, 0)
        else:
            text = "Bad detection"; color = (0, 0, 220)
    else:
        prediction_buffer.clear()

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, 10), (tw+30, th+30), (0,0,0), -1)
    cv2.putText(frame, text, (20, th+18), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, "Press ESC to quit", (20, frame.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
