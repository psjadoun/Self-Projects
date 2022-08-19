import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
img = cv2.imread('OCR_text_detection/Resources/doc.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(pytesseract.image_to_string(img))

# # detecting characters
# hImg, wImg, _ = img.shape
# boxes = pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#     # print(b)
#     b = b.split(" ")
#     x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv2.rectangle(img, (x1, hImg-y1), (x2, hImg-y2), (0, 0, 255), 1)
#     cv2.putText(img, b[0], (x1, hImg-y1+25),
#                 cv2.FONT_HERSHEY_COMPLEX, 0.5, (50, 50, 255), 1)

# # detecting numeric characters
# hImg, wImg, _ = img.shape
# config = r'--oem 3 --psm 6 outputbase digits'
# boxes = pytesseract.image_to_boxes(img, config=config)
# for b in boxes.splitlines():
#     # print(b)
#     b = b.split(" ")
#     x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv2.rectangle(img, (x1, hImg-y1), (x2, hImg-y2), (0, 0, 255), 1)
#     cv2.putText(img, b[0], (x1, hImg-y1+25),
#                 cv2.FONT_HERSHEY_COMPLEX, 0.5, (50, 50, 255), 1)

# detecting words
hImg, wImg, _ = img.shape
boxes = pytesseract.image_to_data(img)
for i, b in enumerate(boxes.splitlines()):
    if i != 0:
        b = b.split()
        if len(b) == 12:
            print(b)
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(img, b[-1], (x, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (50, 50, 255), 1)

# # detecting Numeric words
# hImg, wImg, _ = img.shape
# config = r'--oem 3 --psm 6 outputbase digits'
# boxes = pytesseract.image_to_data(img, config=config)
# for i, b in enumerate(boxes.splitlines()):
#     if i != 0:
#         b = b.split()
#         if len(b) == 12:
#             print(b)
#             x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
#             cv2.putText(img, b[-1], (x, y),
#                         cv2.FONT_HERSHEY_COMPLEX, 0.5, (50, 50, 255), 1)

cv2.imshow("Result", img)
cv2.waitKey(0)
