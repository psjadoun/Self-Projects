from sre_constants import SUCCESS
import cv2
from cv2 import waitKey
from cv2 import COLOR_BGR2RGB
import mediapipe as mp
import time
import HandTrackingModule.HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(10, 150)
detector = htm.handDetector()
while True:
    SUCCESS, img = cap.read()
    detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4])
    cv2.imshow("Image", img)
    if waitKey(1) & 0xff == ord("q"):
        break
