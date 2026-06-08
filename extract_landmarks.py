import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from features import extract_normalized_landmarks, engineer_features, COLUMNS

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATASET_PATH         = r"C:\Users\MADDY\OneDrive\Documents\asl_alphabet_train\asl_alphabet_train"
MAX_IMAGES_PER_CLASS = 1500          # ← raised from 500
OUTPUT_CSV           = "landmarks.csv"

mp_hands = mp.solutions.hands
data, skipped = [], 0

with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                    min_detection_confidence=0.5) as hands:
    for label in sorted(os.listdir(DATASET_PATH)):
        class_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(class_path):
            continue
        for img_name in tqdm(os.listdir(class_path)[:MAX_IMAGES_PER_CLASS],
                             desc=f"Processing {label}"):
            img = cv2.imread(os.path.join(class_path, img_name))
            if img is None:
                skipped += 1; continue
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                skipped += 1; continue
            flat63 = extract_normalized_landmarks(results.multi_hand_landmarks[0])
            if flat63 is None:
                skipped += 1; continue
            row = engineer_features(flat63)
            row.append(label)
            data.append(row)

all_cols = COLUMNS + ["label"]
df = pd.DataFrame(data, columns=all_cols)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone! Rows: {len(df)} | Skipped: {skipped} | Features: {len(COLUMNS)}")
print(df["label"].value_counts().sort_index())