import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
from features import COLUMNS  # ← single source of truth

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Load ──────────────────────────────────────────────────────────────────────
df    = pd.read_csv("landmarks.csv")
print("Class distribution:\n", df["label"].value_counts().sort_index())
print(f"\nTotal samples : {len(df)}")
print(f"Total classes : {df['label'].nunique()}")

X     = df.drop("label", axis=1).values
y_raw = df["label"].values
print(f"Features      : {X.shape[1]}  (must be 84)")
assert X.shape[1] == 84, f"Expected 84 features, got {X.shape[1]}"

# ── Augment: jitter landmark coords to simulate real-webcam noise ─────────────
np.random.seed(42)
noise  = np.random.normal(0, 0.01, X.shape)   # tiny positional noise
X_aug  = X + noise
# Also add flipped version (mirrors left/right hand)
X_flip = X.copy()
X_flip[:, 0::3] = -X_flip[:, 0::3]            # negate all x coords
X      = np.vstack([X, X_aug, X_flip])
y_raw  = np.hstack([y_raw, y_raw, y_raw])
print(f"After augment : {len(X)} samples")

# ── Encode ────────────────────────────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(y_raw)

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Model ─────────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2,
    max_features="sqrt", random_state=42, n_jobs=-1)

svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                   probability=True, random_state=42))
])

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("svm", svm_pipe)],
    voting="soft", weights=[1, 2], n_jobs=-1)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining ensemble (RF + SVM) — takes 2-5 minutes with augmented data...")
ensemble.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = ensemble.predict(X_test)
print(f"\nTest accuracy : {accuracy_score(y_test, preds)*100:.2f}%")
print(classification_report(y_test, preds, target_names=le.classes_))

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble, X, y, cv=cv, n_jobs=-1)
print(f"CV accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(ensemble, "sign_model.pkl")
joblib.dump(le,       "label_encoder.pkl")
print("\nSaved: sign_model.pkl + label_encoder.pkl")
print(f"Model expects : {ensemble.n_features_in_} features")  # must be 84