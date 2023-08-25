from turtle import Screen, Turtle
import cv2
import mediapipe as mp
import time

# Set up Turtle
screen = Screen()
screen.setup(1000, 1000)
t = Turtle("arrow")
t.speed("fastest")  # Set the fastest speed for the Turtle
t.penup()  # Initial pen state is up
t.pensize(5)

# Create a VideoCapture object with lower resolution and MJPEG codec
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Initialize Mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.3)
mpDraw = mp.solutions.drawing_utils

prev_time = time.time()

while True:
    ret, image = cap.read()

    if ret:
        start_time = time.time()
        results = hands.process(image)
        end_time = time.time()
        processing_time = end_time - start_time
        fps = 1 / processing_time

        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        t.penup()  # Reset pen state before processing each frame

        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]  # We are interested in the first hand only
            pointer_coords = ()
            thumb_coords = ()

            for lm_id, lm in enumerate(handLms.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                if lm_id == 4:
                    thumb_coords = (cx, cy)
                if lm_id == 8:
                    pointer_coords = (cx, cy)
                    fx = ((screen.window_width() / image.shape[1]) * cx * -2)
                    fy = ((screen.window_height() / image.shape[0]) * cy * -2)
                    if abs(pointer_coords[0] - thumb_coords[0]) < 60 and abs(pointer_coords[1] - thumb_coords[1]) < 60:
                        t.penup()
                    else:
                        t.pendown()
                    x = fx + screen.window_width()
                    y = fy + screen.window_height()
                    t.setpos(x, y)  # Directly set the position instead of using 'setheading' and 'goto'

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
        else:
            t.clear()

        cv2.imshow("Output", image)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
screen.mainloop()