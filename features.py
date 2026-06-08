"""
features.py — SINGLE SOURCE OF TRUTH
Import this in extract_landmarks.py, trainmodel.py, and app.py.
Never copy-paste these functions again.
"""
import numpy as np

WRIST      = 0
FINGERTIPS = [4, 8, 12, 16, 20]
MCP_JOINTS = [1, 5,  9, 13, 17]
PIP_JOINTS = [2, 6, 10, 14, 18]


def extract_normalized_landmarks(hand_landmarks):
    """
    Accepts a MediaPipe hand_landmarks object (has .landmark list).
    Returns a flat list of 63 floats, or None if scale==0.
    """
    raw     = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    wrist   = raw[0]
    centered = [[p[0]-wrist[0], p[1]-wrist[1], p[2]-wrist[2]] for p in raw]
    all_x   = [p[0] for p in centered]
    all_y   = [p[1] for p in centered]
    scale   = max(max(all_x)-min(all_x), max(all_y)-min(all_y))
    if scale == 0:
        return None
    normed  = [[p[0]/scale, p[1]/scale, p[2]/scale] for p in centered]
    return [v for pt in normed for v in pt]


def extract_normalized_landmarks_from_dicts(raw_list):
    """
    Same as above but accepts a list of dicts [{x,y,z}, ...] from JSON.
    Used in app.py when receiving landmarks from the browser.
    """
    raw     = [[lm["x"], lm["y"], lm["z"]] for lm in raw_list]
    wrist   = raw[0]
    centered = [[p[0]-wrist[0], p[1]-wrist[1], p[2]-wrist[2]] for p in raw]
    all_x   = [p[0] for p in centered]
    all_y   = [p[1] for p in centered]
    scale   = max(max(all_x)-min(all_x), max(all_y)-min(all_y))
    if scale == 0:
        return None
    normed  = [[p[0]/scale, p[1]/scale, p[2]/scale] for p in centered]
    return [v for pt in normed for v in pt]


def engineer_features(flat63):
    """
    Takes the 63-float normalised landmark vector.
    Returns an 84-float feature vector (63 + 21 engineered features).
    ORDER MUST NEVER CHANGE after training.
    """
    pts   = np.array(flat63).reshape(21, 3)
    extra = []

    # 5 tip-to-wrist distances
    wrist = pts[WRIST]
    for tip in FINGERTIPS:
        extra.append(float(np.linalg.norm(pts[tip] - wrist)))

    # 10 tip-to-tip distances
    for i in range(len(FINGERTIPS)):
        for j in range(i+1, len(FINGERTIPS)):
            extra.append(float(np.linalg.norm(pts[FINGERTIPS[i]] - pts[FINGERTIPS[j]])))

    # 5 finger curl angles
    for k in range(5):
        v1 = pts[PIP_JOINTS[k]] - pts[MCP_JOINTS[k]]
        v2 = pts[FINGERTIPS[k]] - pts[PIP_JOINTS[k]]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        extra.append(
            float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
            if n1 > 0 and n2 > 0 else 0.0
        )

    # 1 palm aspect ratio
    all_x, all_y = pts[:, 0], pts[:, 1]
    w = max(all_x) - min(all_x)
    h = max(all_y) - min(all_y)
    extra.append(w / h if h > 0 else 1.0)

    return flat63 + extra  # 63 + 21 = 84


# Column names — identical order used at train AND predict time
BASE_COLS  = [f"{ax}{i}" for i in range(21) for ax in ["x", "y", "z"]]
EXTRA_COLS = (
    [f"tip_wrist_{i}" for i in range(5)] +
    [f"tip_dist_{i}_{j}" for i in range(5) for j in range(i+1, 5)] +
    [f"curl_{i}" for i in range(5)] +
    ["palm_ratio"]
)
COLUMNS = BASE_COLS + EXTRA_COLS  # exactly 84

assert len(COLUMNS) == 84, f"Expected 84 columns, got {len(COLUMNS)}"