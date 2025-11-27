import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Hàm tạo GLOW đẹp
def draw_glow(img, pts, color=(0, 255, 255), size=25):
    glow = np.zeros_like(img)
    for p in pts:
        cv2.circle(glow, p, size, color, -1)
    glow = cv2.GaussianBlur(glow, (55, 55), 35)
    img[:] = cv2.addWeighted(img, 1.0, glow, 0.6, 0)
    return img

# Main
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
) as hands:

    prev = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        small = cv2.resize(frame, (480, 360))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # FPS
        now = time.time()
        fps = 1 / (now - prev) if prev != 0 else 0
        prev = now

        # Draw hands
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                pts = []
                for lm in hand.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    pts.append((x, y))

                # Glow effect
                frame = draw_glow(frame, pts, (0, 255, 255), size=18)

                # Draw connections
                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3)

        cv2.imshow("Hung Nguyen | Developer", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
#Nhấn ESC để thoát