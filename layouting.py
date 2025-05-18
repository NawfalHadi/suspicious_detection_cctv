import cv2
import numpy as np

# Load all 5 video sources
original = cv2.VideoCapture('test/vid1.mp4')
small1 = cv2.VideoCapture('test/vid2.mp4')
small2 = cv2.VideoCapture('test/vid1.mp4')
small3 = cv2.VideoCapture('test/vid2.mp4')
small4 = cv2.VideoCapture('test/vid1.mp4')

# Target sizes for layout
original_width, original_height = 960, 540
small_width, small_height = 480, 270

while True:
    ret0, frame0 = original.read()
    ret1, frame1 = small1.read()
    ret2, frame2 = small2.read()
    ret3, frame3 = small3.read()
    ret4, frame4 = small4.read()

    if not all([ret0, ret1, ret2, ret3, ret4]):
        print("One of the videos ended or failed to load.")
        break

    # Resize each frame to its target size
    frame0 = cv2.resize(frame0, (original_width, original_height))
    frame1 = cv2.resize(frame1, (small_width, small_height))
    frame2 = cv2.resize(frame2, (small_width, small_height))
    frame3 = cv2.resize(frame3, (small_width, small_height))
    frame4 = cv2.resize(frame4, (small_width, small_height))

    # Create two rows of small videos
    top_small_row = np.hstack((frame1, frame2))
    bottom_small_row = np.hstack((frame3, frame4))
    small_grid = np.vstack((top_small_row, bottom_small_row))

    # Combine original on top and the small grid below
    final_layout = np.vstack((frame0, small_grid))

    # Show the result
    cv2.imshow("Custom Video Layout", final_layout)

    # Exit on 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Cleanup
original.release()
small1.release()
small2.release()
small3.release()
small4.release()
cv2.destroyAllWindows()