import cv2
import numpy as np
import os

# importing images

path = "2_Feature_detection_img_classification/ImageQuery/flowers"
images = []
classNames = []
# lists out the names of content inside folder - path
myList = os.listdir(path)
print("Total classes detected: ", len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}')
    images.append(imgCur)
    # list of image names without extension
    classNames.append(os.path.splitext(cl)[0])

orb = cv2.ORB_create(nfeatures=1000)


def findDes(images):
    desList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList


def findID(img, desList, thres=15):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            match = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in match:
                if m.distance < 0.75*n.distance:
                    good.append(m)
            matchList.append(len(good))
        # print(matchList)
    except:
        pass
    if len(matchList) != 0:
        # if max(matchList) > thres:
        finalVal = np.argmax(matchList)
    return finalVal


# cap = cv2.VideoCapture(0)
desList = findDes(images)

# cap = cv2.VideoCapture(0)
# while True:
#     success, img = cap.read()

List2 = os.listdir("2_Feature_detection_img_classification/ImageTrain/flowers")
for imgName in List2:
    img = cv2.imread(
        f'2_Feature_detection_img_classification/ImageTrain/flowers/{imgName}')

    imgOrg = img.copy()

    id = findID(img, desList)
    if id != -1:
        cv2.putText(imgOrg, classNames[id], (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("img", imgOrg)
    if cv2.waitKey(2000) & 0xff == ord('q'):
        break

# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# cv2.imshow("img1", img1)
# cv2.imshow("img2", img2)
