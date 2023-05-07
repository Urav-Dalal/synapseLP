import mediapipe as mp
import cv2

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils 

class HandDetector:
    def __init__(self,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5): #confidence values are default values
        self.hands=mpHands.Hands(max_num_hands=max_num_hands,min_detection_confidence=min_detection_confidence,min_tracking_confidence=min_tracking_confidence)

    def findLandmarks(self,image,handNumber=0,draw=False):
        originalImage = image
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)#converts image from bgr to rgb for mediapipe
        results = self.hands.process(image)
        landMarkList = []

        if(results.multi_hand_landmarks): #this has the landmarks of our hand,21 landmarks
            hands = results.multi_hand_landmarks[handNumber]
        
        print(results.multi_hand_landmarks)
        for id,landmark in enumerate(hands.landmark):
                imgH,imgW,imgC = originalImage.shape

                xPos,yPos = int(landmark.x*imgW),int(landmark.y*imgH)

                landMarkList.append([id,xPos,yPos])

        if draw:
                mpDraw.draw_landmarks(originalImage,hands,mpHands.HAND_CONNECTIONS)

        return landMarkList


