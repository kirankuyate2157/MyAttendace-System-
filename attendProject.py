import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime, date


path = 'database'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)
for Tdata in myList:  # accessing data of stored in data directory
    currentImg = cv2.imread(f'{path}/{Tdata}')
    images.append(currentImg)
    classNames.append(os.path.splitext(Tdata)[0])


def findEncodings(images):
    encodList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodList.append(encode)
    return encodList


def makeAttendace(name):  # attendances  updating in csv file
    with open('attendace1.csv', 'r+') as f:
        MyDataList = f.readlines()
        nameList = []
        for line in MyDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            dateString = now.strftime("%d/%m/%Y")
            f.writelines(f'\n{name},{dateString},{dtString}')  # name,date,time


encodeListKnown = findEncodings(images)  # encoding img into array format like
print("encoding is completed !!!!")

# hey i`m kiran kuyate if incase your camera is not working just change value  0 to number of camera in your system have.. since i have  0 for my laptop camera and second 1 of external camera..
VideoCap = cv2.VideoCapture(1)  # video capture from webcam

while True:
    isTrue, img = VideoCap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(
        imgs, facesCurFrame)  # encoding frames

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(
            encodeListKnown, encodeFace)  # comparing faces data
        # on comparing matching data how similar it is,if it's distancs is less then then good it's maching if not then not well matches ...
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            makeAttendace(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # creating frame where face is detected
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 23), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 23), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break
    cv2.imshow("webCam", img)
    cv2.waitKey(1)
videoCap.release()
cv2.destroyAllWindows()

# for stoping program on terminal press 'd' or Ctr + C
