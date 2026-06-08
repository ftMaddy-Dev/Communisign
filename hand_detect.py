import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            # Detect index finger up
            tip = hand.landmark[8]
            joint = hand.landmark[6]

            if tip.y < joint.y:
                cv2.putText(
                    frame,
                    "INDEX UP",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            # Finger counting
            count = 0

            tips = [8, 12, 16, 20]

            for tip_id in tips:
                if hand.landmark[tip_id].y < hand.landmark[tip_id - 2].y:
                    count += 1

            cv2.putText(
                frame,
                f"Fingers: {count}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # Draw landmark numbers
            for id, lm in enumerate(hand.landmark):

                h, w, c = frame.shape

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.putText(
                    frame,
                    str(id),
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()