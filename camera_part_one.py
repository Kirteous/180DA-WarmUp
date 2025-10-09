import numpy as np
import cv2

cap = cv2.VideoCapture(1)	

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

     # define range of blue color in HSV
    lower_level = np.array([110,50,50])
    upper_level = np.array([135,255,255])
    
    # Threshold the HSV image to get only pink colors
    mask = cv2.inRange(hsv, lower_level, upper_level)
 
    contours0, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours0:
        c = max(contours0, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()