"""
Python script to test camera
"""

import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        exit()

    cv2.imshow(f"Camera {i}", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()