import random
import cv2
import mediapipe as mp
import util
import pyautogui
import numpy
from pynput.mouse import Button,Controller
mouse = Controller() 


screen_width,screen_height = pyautogui.size()
#initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence =0.6,
    max_num_hands=1
)


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None  


def move_mouse(index_finger_tip):
    if index_finger_tip  is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int (index_finger_tip.y * screen_width)
        pyautogui.moveTo(x,y)
        
 #left click       
def is_left_click(landmark_list,thumb_index_dist):
    return (
            util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8])<50 and 
            util.get_angle(landmark_list[9],landmark_list[10],landmark_list[12])>90 and
            thumb_index_dist>50
    )
#right click       
def is_right_click(landmark_list,thumb_index_dist):
    return (
            util.get_angle(landmark_list[9],landmark_list[10],landmark_list[12])<50 and 
            util.get_angle (landmark_list[5],landmark_list[6],landmark_list[8])>90 and
            thumb_index_dist >50
    )
    
#double click       
def is_double_click(landmark_list,thumb_index_dist):
    return (
            util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8])<50 and 
            util.get_angle(landmark_list[9],landmark_list[10],landmark_list[12])<50 and
            thumb_index_dist>50
    )
    
#screenshot     
def is_screenshot(landmark_list,thumb_index_dist):
    return (
            util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8])<50 and 
            util.get_angle(landmark_list[9],landmark_list[10],landmark_list[12])<50 and
            thumb_index_dist<50
    )
    
def detect_gestures (frame,landmark_list,processed):
    if len(landmark_list)>=21:
        index_finger_tip = find_finger_tip (processed)
        #print(index_finger_tip)
        thumb_index_dist = util.get_distance([landmark_list[4],landmark_list[5]])
        
        if thumb_index_dist <50 and util.get_angle (landmark_list[5],landmark_list[6],landmark_list[8])>90:
            move_mouse(index_finger_tip)
        elif is_left_click(landmark_list,thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame,"leftclick",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        elif is_right_click(landmark_list,thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame,"rightclick",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        elif is_double_click(landmark_list,thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame,"doubleclick",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        elif is_screenshot(landmark_list,thumb_index_dist):
            im1 = pyautogui.screenshot()
            label = random.randint(1,1000)
            im1.save ( 'my_screenshot_{label}.png')
            cv2.putText(frame,"Screenshot",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    
    
def main():
    draw =mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)
            landmarks_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks (frame,hand_landmarks,mpHands.HAND_CONNECTIONS)            
                for lm in hand_landmarks.landmark:
                    landmarks_list.append((lm.x,lm.y))
                
            detect_gestures(frame,landmarks_list,processed)
            
            cv2.imshow ('frame',frame)
            if cv2.waitKey (1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    
if __name__== '__main__':
    main()  