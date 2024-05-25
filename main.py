import cv2
import numpy as np
vid = cv2.VideoCapture(0)
isRunning = True
height, width, _ = vid.shape
scale_percent = 50
new_width = int(width * scale_percent / 100)
new_height = int(height * scale_percent / 100)
    dim = (new_width, new_height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
while(isRunning):
    ret, frame = vid.read()
    cv2.imshow('LEARN ASL',frame)
    if(cv2.waitKey(1) and 0xFF == ord('q')):
        break
vid.release()
cv2.destroyAllWindows()




    # Resize the frame
    

    # Display the resized frame
cv2.imshow('Resized Frame', resized)
