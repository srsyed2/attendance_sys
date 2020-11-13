# -*- coding: utf-8 -*-
"""
@author: Group 09 - MACHINE VISION UTM FKE
"""

#The required libraries are imported here

import tkinter as  TK
from tkinter import Message,Text
import cv2,os
import csv
import numpy as NPy
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time


# Code for Graphics user interface

window =TK.Tk()
window.title("Attendance System")

dialog_title = 'TERMINATE PROGRAM??'
dialog_text = 'Sure?'

window.geometry('1380x780')
window.configure(background='Moccasin')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = TK.Label(window, text="Student Attendance System- Machine Vision" ,bg="White"  ,fg="Black"  ,width=50  ,height=2,font=('Helvetica', 34, 'italic bold underline'))
message.place(x=65, y=15)

lbl = TK.Label(window, text="Matric No.",width=20  ,height=2  ,fg="black"  ,bg="Light Blue" ,font=('Helvetica', 16, ' bold ') )
lbl.place(x=400, y=200)

txt = TK.Entry(window,width=20  ,bg="Light Blue" ,fg="black",font=('Helvetica', 16, ' bold '))
txt.place(x=700, y=215)

lbl2 = TK.Label(window, text="Name" ,width=20  ,fg="black"  ,bg="Light Blue"   ,height=2 ,font=('Helvetica', 16, ' bold '))
lbl2.place(x=400, y=300)

txt2 =TK.Entry(window,width=20  ,bg="Light Blue",fg="black",font=('Helvetica', 16, ' bold ')  )
txt2.place(x=700, y=315)

lbl3 = TK.Label(window, text="Update : ",width=20  ,fg="black"  ,bg="Light Blue"  ,height=2 ,font=('Helvetica', 16, ' bold '))
lbl3.place(x=400, y=400)

message = TK.Label(window, text="" ,bg="Light Blue"  ,fg="black"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 16, ' bold '))
message.place(x=700, y=400)

lbl3 = TK.Label(window, text="Attendance : ",width=20  ,fg="black"  ,bg="Light Blue"  ,height=2 ,font=('times', 16, ' bold  underline'))
lbl3.place(x=400, y=650)


message2 = TK.Label(window, text="" ,fg="red"   ,bg="Light Blue",activeforeground = "green",width=30  ,height=2  ,font=('times', 16, ' bold '))
message2.place(x=700, y=650)



def CLR(): #Clear Textbox button
    txt.delete(0, 'end')
    res = ""
    message.configure(text= res)

    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


def is_number(s): #for Verification of ID no.
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


#this function is responsible for detection of a face from the input image
def InpImg():
    Id = (txt.get())
    name = (txt2.get())
    if (is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name
        row = [Id, name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if (is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if (name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text=res)



#This function trains the image that have been taken of the student for registration.
def TrainImages():

    recognizer = cv2.face_LBPHFaceRecognizer.create() #recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, NPy.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Training Successful!" #+",".join(str(f) for f in Id)
    message.configure(text= res)


#After image is taken this function assigns ID and name to the unique face and stores in the database

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #print(imagePaths)

    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=NPy.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces,Ids


def TrackImages(): #This function helps to extract features and compare with database.
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    haarcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(haarcascadePath);
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)

    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]

            else:
                Id='Unknown'
                tt=str(Id)
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
        cv2.imshow('im',im)
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)

#code for the buttons used in GUI
clearButton = TK.Button(window, text="Clear", command=CLR ,fg="blue"  ,bg="light green"  ,width=16  ,height=2 ,activebackground = "Red" ,font=('Halvetica', 17, ' bold '))
clearButton.place(x=980, y=220)

takeImg = TK.Button(window, text="Store Image", command=InpImg  ,fg="blue"  ,bg="white"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 16, ' bold '))
takeImg.place(x=200, y=500)
trainImg = TK.Button(window, text="Train ", command=TrainImages  ,fg="blue"  ,bg="white"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 16, ' bold '))
trainImg.place(x=500, y=500)
trackImg = TK.Button(window, text="Take Attendance", command=TrackImages  ,fg="blue"  ,bg="white"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 16, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = TK.Button(window, text="Quit", command=window.destroy  ,fg="red"  ,bg="white"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 16, ' bold '))
quitWindow.place(x=1100, y=500)
copyWrite = TK.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.insert("insert", "")
copyWrite.configure(state="disabled",fg="red"  )
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750)



window.mainloop()