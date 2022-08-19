import cv2
import numpy as np
import HandTrackingModule.HandTrackingModule as htm

wCam = 640
hCam = 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionCon=0.7)
boxes = np.array([[100, 100, 300, 300], [300, 100, 500, 300]])

while True:
    success, img = cap.read()

    # Draw Transparncy
    imgNew = np.zeros_like(img, np.uint8)
    for box in boxes:
        cv2.rectangle(imgNew, box[:2], box[2:], (255, 0, 255), cv2.FILLED)

    detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[4][1:]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        length = np.sqrt((x1-x2)**2+(y1-y2)**2)
        # print(length)
        if length < 50:
            for i, box in enumerate(boxes):
                if box[0] < x1 < box[2] and box[0] < x2 < box[2] and box[1] < y1 < box[3] and box[1] < y2 < box[3]:
                    dx, dy = box[2]-box[0], box[3] - box[1]
                    boxes[i] = [(2*cx-dx)//2, (2*cy-dy)//2,
                                (2*cx+dx)//2, (2*cy+dy)//2]
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("video", out)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
