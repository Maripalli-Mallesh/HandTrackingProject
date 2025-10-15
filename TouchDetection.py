import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Draw circles on tips
            cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (index_x, index_y), 8, (0, 255, 255), cv2.FILLED)

            # Draw line between tips
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 255, 0), 3)

            # Compute distance
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

            # Choose text and color
            if distance < 40:
                text = "Touched"
                color = (0, 255, 0)
            else:
                text = "Not Touched"
                color = (0, 0, 255)

            # ✅ Display text near that specific hand
            cx, cy = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2
            cv2.putText(frame, text, (cx - 50, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    cv2.imshow("Two Hand Thumb & Index Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
