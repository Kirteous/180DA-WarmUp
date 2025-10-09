"""used Kmeans and dominant color tutorial"""

import numpy as np
import cv2
from sklearn.cluster import KMeans

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    img_height,img_width,_ = frame.shape
    track_height = img_height//4
    track_width = img_width//4
    xl = img_width//2 - track_width//2
    xr = xl + track_width
    yb = img_height//2 - track_height//2
    yt = yb + track_height

    region_of_interest = frame[yb:yt, xl:xr]
    array = region_of_interest.reshape(-1,3)
    
    kmeans = KMeans(n_clusters=1, n_init="auto", random_state=42)
    kmeans.fit(array)
    dom_color = kmeans.cluster_centers_[0].astype(np.uint8)

    cv2.rectangle(frame, (xl, yb), (xr, yt), (0, 255, 0), 2)
    template = np.zeros((100,100,3), dtype = np.uint8)
    template[:] = dom_color

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow("Dominant Color", template)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()