import cv2 as cv
import HandTrackingModule as track
import os

wCam, hCam = 1080, 720
cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath = "./Image"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    img = cv.imread(f"{folderPath}/{imPath}")
    overlayList.append(img)

detector = track.HandDetector(maxHands=1, detectionCon=0.8)
tipIds = [4,8,12,16,20]

while True:
    ret, frame = cap.read()
    img = detector.findHands(frame, draw=False)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Four fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)
        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

    cv.imshow("Image", img)


    key = cv.waitKey(1)
    if key == ord('q'):
        break