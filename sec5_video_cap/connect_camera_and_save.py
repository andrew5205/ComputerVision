
import cv2 


cap = cv2.VideoCapture(0)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 1080
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))         # 1080 



""" writer object - cv2.VideoWriter_fourcc""" 
# Windows -- *'DIVX'
# MAC Linux -- *'XVID'
writer = cv2.VideoWriter('fileName.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))



while True:
    
    ret, frame = cap.read()
    
    # Operations 
    writer.write(frame)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', gray)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
writer.release()

cv2.destroyAllWindows











