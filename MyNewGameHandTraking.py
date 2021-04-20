import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

cTime = 0
pTime = 0

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.FindHands(img)
    LmList = detector.FindPosition(img, False)
    if len(LmList) != 0:
        print(LmList[4])


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

    cv2.imshow("Create by Bexruz Nutfilloyev", img)
    cv2.waitKey(1)
