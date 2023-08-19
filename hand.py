import cv2
import mediapipe as mp
import time

# Configure OpenCV to use the native M1 optimized libraries
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# Create a VideoCapture object with lower resolution and MJPEG codec
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Initialize Mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, 
min_detection_confidence=0.5, min_tracking_confidence=0.3)
mpDraw = mp.solutions.drawing_utils

prev_time = 0

while True:
    ret, image = cap.read()

    if ret:
        # Process the frame with mediapipe hands
        start_time = time.time()
        results = hands.process(image)
        end_time = time.time()

        processing_time = end_time - start_time

        fps = 1 / processing_time
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), 
cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for lm in handLms.landmark:
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 5, (255, 0, 255), 
cv2.FILLED)
        
            mpDraw.draw_landmarks(image, handLms, 
mpHands.HAND_CONNECTIONS)

        cv2.imshow("Output", image)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

