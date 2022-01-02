# USAGE
# python recognize_video.py --detector face_detection_model \
#   --embedding-model openface_nn4.small2.v1.t7 \
#   --recognizer output/recognizer.pickle \
#   --le output/le.pickle

# import the necessary packages

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import RPi.GPIO as GPIO
import time
import spidev
from lib_nrf24 import NRF24
import board
import digitalio
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)
GPIO.setup(20, GPIO.OUT)
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)
 
IN1 = 6
IN2 = 13
IN3 = 19
IN4 = 26

time = 0.001

GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

locked = True

# set GPIO mode
GPIO.setmode(GPIO.BCM)

# set pipe address
pipes = [[0xE0, 0xE0, 0xF1, 0xF1, 0xE0], [0xF1, 0xF1, 0xF0, 0xF0, 0xE0]]

# create the radio
radio = NRF24(GPIO, spidev.SpiDev())

# start the radio
radio.begin(0, 25)

# initialize the radio
radio.setPayloadSize(32)
radio.setChannel(0x76)
radio.setDataRate(NRF24.BR_1MBPS)
radio.setPALevel(NRF24.PA_MIN)

radio.setAutoAck(True);
radio.enableDynamicPayloads();
radio.enableAckPayload();

# open pipes
radio.openWritingPipe(pipes[0])
#radio.printDetails()

# init sendMessage
#sendMessage = list("Hi..Arduino UNO");

def Step1():
    GPIO.output(IN4, True)
    sleep(time)
    GPIO.output(IN4, False)

def Step2():
    GPIO.output(IN4, True)
    GPIO.output(IN3, True)
    sleep(time)
    GPIO.output(IN4, False)
    GPIO.output(IN3, False)

def Step3():
    GPIO.output(IN3, True)
    sleep(time)
    GPIO.output(IN3, False)

def Step4():
    GPIO.output(IN2, True)
    GPIO.output(IN3, True)
    sleep(time)
    GPIO.output(IN2, False)
    GPIO.output(IN3, False)

def Step5():
    GPIO.output(IN2, True)
    sleep(time)
    GPIO.output(IN2, False)

def Step6():
    GPIO.output(IN1, True)
    GPIO.output(IN2, True)
    sleep(time)
    GPIO.output(IN1, False)
    GPIO.output(IN2, False)

def Step7():
    GPIO.output(IN1, True)
    sleep(time)
    GPIO.output(IN1, False)

def Step8():
    GPIO.output(IN4, True)
    GPIO.output(IN1, True)
    sleep(time)
    GPIO.output(IN4, False)
    GPIO.output(IN1, False)

# Umdrehung links herum  
def lock(step): 
    for i in range (step):   
        #os.system('clear') # verlangsamt die Bewegung des Motors zu sehr. 
        Step1()
        Step2()
        Step3()
        Step4()
        Step5()
        Step6()
        Step7()
        Step8()  
        #print "Step left: ",i

# Umdrehung rechts herum        
def unlock(step):
    for i in range (step):
        #os.system('clear') # verlangsamt die Bewegung des Motors zu sehr.  
        Step8()
        Step7()
        Step6()
        Step5()
        Step4()
        Step3()
        Step2()
        Step1()  
        #print "Step right: ",i
                                        
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
    help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
    help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
    help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
    help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
    lockState = GPIO.input(23)
    unlockState = GPIO.input(24)
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            if name == "david" and locked == True:
                GPIO.output(20, GPIO.HIGH)
                GPIO.output(21, GPIO.LOW)
                unlock(512)
                GPIO.output(20, GPIO.LOW)
                GPIO.output(21, GPIO.HIGH)
                locked = False
                #user_steps = input("How many steps backwards? ")
                #backwards(int(user_delay) / 1000.0, int(user_steps))
                
    if lockState == False and locked == False:
        print("Locked!")
        locked = True
        lock(512)
        GPIO.output(20, GPIO.HIGH)
        GPIO.output(21, GPIO.LOW)
                
    if unlockState == False and locked == True:
        print("Unlocked!")
        locked = False
        unlock(512)
        GPIO.output(20, GPIO.LOW)
        GPIO.output(21, GPIO.HIGH)

    #print(sendMessage[0])
    # update messages
    #while True:
        #print("okcool")
        #start = time.time()
        #radio.write(sendMessage)
        #print("Sent the message: {}".format(sendMessage))
        #radio.startListening()
        
    #while not radio.available(0):
        #print("oof")
        #time.sleep(1/100)
        #if time.time() - start > 2:
            #print("Timed out.")
            #break
        
        #radio.stopListening()
        #time.sleep(3)

    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()