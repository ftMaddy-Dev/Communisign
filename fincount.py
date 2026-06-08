import cv2
import mediapipe as mp
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip so it acts like a mirror — fixes thumb left/right logic
    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            lm    = hand_landmarks.landmark
            count = 0

            # ── Thumb ────────────────────────────────────────────────────
            # Thumb moves sideways (x-axis), not up/down.
            # For a RIGHT hand (after flip): tip.x < ip.x  means extended.
            # For a LEFT  hand             : tip.x > ip.x  means extended.
            label = handedness.classification[0].label   # "Right" or "Left"
            if label == "Right":
                if lm[4].x < lm[3].x:
                    count += 1
            else:
                if lm[4].x > lm[3].x:
                    count += 1

            # ── Four fingers (index → pinky) ──────────────────────────────
            # Tip is ABOVE its PIP joint (lower y value) when finger is up.
            for tip_id in [8, 12, 16, 20]:
                if lm[tip_id].y < lm[tip_id - 2].y:
                    count += 1

            # ── Draw skeleton ─────────────────────────────────────────────
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ── Label above the wrist ─────────────────────────────────────
            wrist_x = int(lm[0].x * frame.shape[1])
            wrist_y = int(lm[0].y * frame.shape[0])

            cv2.putText(
                frame,
                f"Fingers: {count}",
                (wrist_x - 40, wrist_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2
            )

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()