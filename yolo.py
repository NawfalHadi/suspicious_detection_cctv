import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
import tensorflow
import time

pred_model = tensorflow.keras.models.load_model('best_model.h5')
data_model = YOLO('yolo11n-pose.pt')

label = ['Normal', 'Suspicious']

start_time = time.time()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f} seconds\n")
    if total_time < 5:
        continue
    
    # Recolor Feed
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = data_model(image)

    plotted_img = results[0].plot()



    for result in results:
        # image = data_model(image)

        kp_array = result.keypoints.xy.cpu().numpy()[0]
        flat_keypoints = kp_array.flatten().tolist()

        # X = pd.DataFrame([row])
        # X.columns = range(X.shape[1])
        X = np.array(flat_keypoints).reshape(1, -1)
        y_pred = pred_model.predict(X)
        y_pred = y_pred[0][0]

        cv2.putText(plotted_img, f'Pred: {label[int(np.round(y_pred))]}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 2)
        cv2.putText(plotted_img, f'Sus Score: {y_pred}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 2)
        
    image = cv2.cvtColor(plotted_img, cv2.COLOR_RGB2BGR)
    
                    
    cv2.imshow('Raw Webcam Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()