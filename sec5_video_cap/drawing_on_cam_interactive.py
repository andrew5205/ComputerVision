import cv2 
# cap = cv2.VideoCapture(0)

# 1. set global variable coordinate 
# 2. drawing action based on coordinate 
# 3. callback function 
#     3.1 reset logic 
# 4. connect callback 




""" Global variables """
pt1 = (0,0)
pt2 = (0,0)
topLeft_clicked = False 
botRight_clicked = False 


""" callback function """
def draw_rectangle(event, x, y, flags, param):
    
    global pt1, pt2, topLeft_clicked, botRight_clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print("event: ", event)
        
        """ Reset rectangle """
        # topLeft_clicked, botRight_clicked => bool 
        if topLeft_clicked and botRight_clicked:
            pt1 = (0,0)
            pt2 = (0,0)
            topLeft_clicked = False 
            botRight_clicked = False 
            
        if topLeft_clicked == False:
            pt1 = (x,y)
            topLeft_clicked = True
            
        elif botRight_clicked == False:
            pt2 = (x,y)
            botRight_clicked = True




""" connect to callback """
cap = cv2.VideoCapture(0)

cv2.namedWindow('Test')
cv2.setMouseCallback('Test', draw_rectangle)




while True:
    
    ret, frame = cap.read()


    """ Drawing on frame base off gloable variables""" 
    if topLeft_clicked == True:
        # mount the fisrt click point in circle
        cv2.circle(frame, center=pt1, radius=5, color=(0,255,0), thickness=3)
        print(pt1)
        
    if topLeft_clicked and botRight_clicked:
        cv2.rectangle(frame, pt1, pt2, color=(0,0,255), thickness=3)
        print(pt1, pt2)




    cv2.imshow('Test', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break 
    
cap.release()
cv2.destroyAllWindows()












