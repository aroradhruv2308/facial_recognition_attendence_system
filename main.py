import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
#now here we have to traverse through known faces 

#first we will import the images and for that we are using os 
path = 'image_basic'
known_faces = []
known_names = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    known_faces.append(curImg)
    known_names.append(os.path.splitext(cl)[0])


#now we need encodings of each image so we will make function for that
def findEncodings(known_faces):
    encodeList = []
    for img in known_faces:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#convert to RGB
        encode = face_recognition.face_encodings(img)[0]#find the encodings
        encodeList.append(encode)
    return encodeList
# we have to create the Attendence new file of extention .csv which is comma separated values

def markAttendance(name):
	with open('Attendance.csv','r+') as f:
		myDataList = f.readlines()
		nameList = []
		for line in myDataList:
			entry = line.split(',')
			nameList.append(entry[0])
		if name not in nameList:
			now = datetime.now()
			dtString = now.strftime('%H:%M:%S')
			f.writelines(f'\n{name},{dtString}')
			



encodeListKnown = findEncodings(known_faces)

#now this will open the web cam 0 is the id here
cap = cv2.VideoCapture(0)
#we have to capture each frame for that while loop
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    #we might have multiple images in the frame so for that we need the locations of each faces
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)#now we  will use the another function in face reco to compare encodings of web cam and training images
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if(matches[matchIndex]):
        	name = known_names[matchIndex].upper()
        	print(name)
        	y1,x2,y2,x1 = faceLoc
        	y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        	cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        	cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        	cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        	markAttendance(name)
       



    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


           	







