
import cv2

img = cv2.imread('00-puppy.jpg')
# print(type(img))                # <class 'numpy.ndarray'>

while True:
    
    cv2.imshow('Puppy', img)

    # # if we waited 1 ms, and pressed Esc key(27) 
    # # if cv2.waitkey(1) & 0b11111111 == 27:
    # if cv2.waitKey(1) & 0xFF == 27:
    #     break
    
    
    # ord('q') -> wait for q to be pressed
    # if cv2.waitKey(0) & 0b11111111 == ord('q'):
    #
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    
    
cv2.destroyAllWindows()























