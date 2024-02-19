import cv2
import numpy as np

lowerBound=np.array([0,0,0])
upperBound=np.array([0,0,255])

cam= cv2.VideoCapture("vid.mp4")
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

font=cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img=cam.read()
    img=cv2.resize(img,(854,480))

    #convert BGR to HSV
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # create the Mask
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    #morphology
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    
    conts,h=cv2.findContours(maskClose.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    cv2.drawContours(img,conts,-1,(255,0,0),3)
    for i in range(len(conts)):
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        cv2.putText(img, str(i+1),(x,y+h),font,1,(0,255,255))
    cv2.imshow("camera",img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()