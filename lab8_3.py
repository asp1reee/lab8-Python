import cv2
import numpy as np


def track_marker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10,
                               maxRadius=100)

    height, width = frame.shape[:2]
    square_size = 200
    square_center = (width // 2, height // 2)
    top_left = (square_center[0] - square_size // 2, square_center[1] - square_size // 2)
    bottom_right = (square_center[0] + square_size // 2, square_center[1] + square_size // 2)

    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            if (top_left[0] < x < bottom_right[0]) and (top_left[1] < y < bottom_right[1]):
                cv2.putText(frame, "Marker inside the square", (x - 50, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)

    return frame


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить изображение")
        break
    processed_frame = track_marker(frame)
    cv2.imshow('Marker Tracking', processed_frame)
    # Прерываем при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
