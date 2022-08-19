from sre_constants import SUCCESS
import cv2
from cv2 import waitKey
from cv2 import COLOR_BGR2RGB
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(10, 150)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    SUCCESS, img = cap.read()
    imgRGB = cv2.cvtColor(img, COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            # lm includes x,y,z in fractions
            for id, lm in enumerate(handLMS.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)
    cv2.imshow("Image", img)
    if waitKey(1) & 0xff == ord("q"):
        break
