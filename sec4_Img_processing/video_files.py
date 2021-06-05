
import cv2
import time

cap = cv2.VideoCapture('hand_move.mp4')

if cap.isOpened() == False:
# if not cap.isOpened():
    print('ERROR File Not Found or Codec Used!')


while cap.isOpened():
    
    ret, frame = cap.read()

    if ret == True:
        

        """ 
        writer FP FPS - match with FP setup in writer
        time.sleep(1/FP) 
        """

        # writer 1/20 FPS
        time.sleep(1/20)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
        
        
cap.release()
cv2.destroyAllWindows()
    
    






