import cv2 # OpenCV
import time
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0) #Web camera

cap.set(3, wCam)
cap.set(4, hCam)

overlayList = []

imageList = (
'HandImages/1.jpg',
'HandImages/2.jpg',
'HandImages/3.jpg',
'HandImages/4.jpg',
'HandImages/5.jpg',
'HandImages/6.jpg'
)

for imPath in imageList:
    image = cv2.imread(imPath)
    print(imPath)
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]

        # Son va to'rtburchak
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    # Speed Test
    cTime = time.time()
    fps = 1 / (cTime - pTime) # FPS - frame per second
    pTime = cTime

    #ekranga chiqarish
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)