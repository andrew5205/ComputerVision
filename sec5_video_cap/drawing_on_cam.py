
import cv2 


cap = cv2.VideoCapture(0)


""" set coordinates """ 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Top left corner 
x = width // 2 
y = height // 2

# width & height of Rectangle 
w = width // 4 
h = height // 4 

# bottom right x + width, y + height 





while True:
    
    ret, frame = cap.read()

    # (x,y) (x+w, y+h) => object detection 
    cv2.rectangle(frame, (x,y), (x+w, y+h), color=(0,0,255), thickness=4)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break 
    
cap.release()
cv2.destroyAllWindows()












