import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


###
pTime = 0
###

cap = cv2.VideoCapture(0)
detector = htm.HandDetector(detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.getVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.FindHands(img)
    lmList = detector.FindPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1 + y2)//2

        cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        print(length)

        vol = np.interp(length, [50, 300], minVol, maxVol)
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [ 0, 100])
        print(vol)

        volume.setMasterVolumelevel(vol, None)

        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{int(volPer)}%", (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 4)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.imshow("Bexruz Nutfilloyev", img)
    cv2.waitKey(1)