
#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import numpy as np
import argparse
# Makes the box in image.

def highlightFace(net, frame, conf_threshold=0.6):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]

    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes
# Generic Text renderer
def debugText(resultImg, text, position):
    cv2.putText(resultImg, text, position,cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA )

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEWHEEL:
        # print(flags)
        frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        if flags > 0:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame+60)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame-60)


# Sets up the command line arguments. 
parser=argparse.ArgumentParser()
parser.add_argument('--image')
parser.add_argument('--frame')
args=parser.parse_args()

MODELROOTFOLDER = "Models/"

# List of Models
faceProto= MODELROOTFOLDER+"opencv_face_detector.pbtxt"
faceModel= MODELROOTFOLDER+"opencv_face_detector_uint8.pb"
ageProto= MODELROOTFOLDER+"age_deploy.prototxt"
ageModel= MODELROOTFOLDER+"age_net.caffemodel"
genderProto= MODELROOTFOLDER+"gender_deploy.prototxt"
genderModel= MODELROOTFOLDER+"gender_net.caffemodel"

# No clue on where the model mean values were calculated from.
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# Partitian of age and gender
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

# Set up Neural Network of face detection and gender detection
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# If no file use webcam
video=cv2.VideoCapture(args.image if args.image else 0)
padding=20

frame = int(args.frame if args.frame else 0)
print("Starting at frame "+str(frame))
video.set(cv2.CAP_PROP_POS_FRAMES, frame)

play = 1

windowName = 'AgeDetector'

img = 255 * (np.ones((400, 400, 3), dtype=np.uint8))
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.imshow(windowName, img)
# Set up mouse callback
cv2.setMouseCallback(windowName, mouse_event)

while True:
    # Progress in video by frame.
    hasFrame,frame=video.read()
    # Quit if end of video
    if not hasFrame:
        break
    key = cv2.waitKey(play)
    if key == 32:
        play = 1-play
    if cv2.getWindowProperty(windowName,cv2.WND_PROP_VISIBLE) < 1:
        break
    # get face detection results
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    # Label pause
    if play == 0:
        debugText(resultImg, "Paused", (100,150))
    debugText(resultImg, "Frame: "+str(video.get(cv2.CAP_PROP_POS_FRAMES)), (100,200))
        
    # Report is there is face detected
    if not faceBoxes:
        debugText(resultImg, "No Face Detected", (100,100))
    else:
        debugText(resultImg, "Face Detected: "+ str(len(faceBoxes)), (100,100))
        # Render Boxes
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),
                        max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
            # # These uses neural networks to regress the gender of the facs detected in the image.
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            # genderNet.setInput(blob)
            # genderPreds=genderNet.forward()
            # gender=genderList[genderPreds[0].argmax()]
            # These uses neural networks to regress the age of the facs detected in the image.
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            # cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)    
            cv2.putText(resultImg, f'{age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)    
    # Render Image
    cv2.imshow(windowName, resultImg)


cv2.destroyAllWindows()
print("Closed")