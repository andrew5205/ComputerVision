
import numpy as np 
import cv2


""" Variables """
# True when mouse buttom down, False when mouse buttom up  
drawing = False 
ix = -1
iy = -1




""" Function """
def draw_retangle(event, x, y, flags, params):
    
    # make these var global since we keep checking it
    global ix, iy, drawing 
    
    # print('event: ', event)
    # case: start drawing 
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True 
        ix, iy = x, y
        print('start event: ', event)
        print(x,y)
        
    # case: drag & moving ==> cont making rectangle
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, (ix,iy), (x,y), (0,255,0), -1)
        print('moving event: ', event)
        print(x,y)
    
    # case: end of drawing 
    elif event == cv2.EVENT_LBUTTONUP:
        drawing == False 
        cv2.rectangle(img, (ix,iy), (x,y), (0,255,0), -1)
        print('end event: ', event)
        print(x,y)
        
    




""" Showing Image """
img = np.zeros((512,512,3))

cv2.namedWindow(winname="my_window")

cv2.setMouseCallback('my_window', draw_retangle)


while True:
    cv2.imshow('my_window', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()














