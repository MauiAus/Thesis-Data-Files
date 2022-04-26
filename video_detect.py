# import the necessary packages
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os

try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (512, 512),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (128, 128))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))


    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=16)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

def detect_video(model_path,video_save_path,flag,model_name):
    # load our serialized face detector model from disk
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    backSub = cv2.createBackgroundSubtractorKNN()

    # load the face mask detector model from disk
    pre_start = time.time()
    #maskNet = load_model("saved_model/ResnetV2")
    maskNet = load_model(model_path)
    pre_process = round(time.time() - pre_start,2)
    # initialize the video stream
    print("[INFO] starting video stream...")
    #vs = VideoStream(src=0).start()
    #vs = VideoStream('video_test_side.m4v').start()
    vs = cv2.VideoCapture('video_test_side.m4v')
    fps = FPS().start()

    #out = cv2.VideoWriter("resnet_chest_600.avi",cv2.VideoWriter_fourcc(*'XVID'), 60,(600,337))#600,337
    out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'XVID'), 60, (600, 337))  # 600,337

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        ret, frame = vs.read()
        try:
            frame = imutils.resize(frame, width=600)#600
        except:
            break
        start = time.time()

        end = 1

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        try:
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        except:
            break
        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            #(improperMask,mask, withoutMask) = pred
            (withoutMask,improperMask,mask) = pred
            #["NFMD", "IFMD", "CFMD"]
            (CFMD,IFMD,NFMD) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            if CFMD > IFMD:
                label = "Mask"
                color = (0, 255, 0)
            elif IFMD > NFMD:
                label = "Improper Mask"
                color = (255, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)
            '''
            if NFMD > CFMD and NFMD > IFMD:
                label = "No Mask"
                color = (0, 0, 255)
            elif IFMD > CFMD and IFMD > NFMD:
                label = "Improper Mask"
                color = (255, 255, 0)
            elif CFMD > IFMD and CFMD > NFMD:
                label = "Mask"
                color = (0, 255, 0)
            '''
            '''
            if withoutMask > mask and withoutMask > improperMask:
                label = "No Mask"
                color = (0, 0, 255)
            elif improperMask > mask and improperMask > withoutMask:
                label = "Improper Mask"
                color = (255, 255, 0)
            elif mask > improperMask and mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
            '''

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(improperMask, mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            end = time.time()


        #end = time.time()

        seconds = end - start

        if seconds == 0:
            seconds = 0.01
        frps = 1 / seconds



        cv2.putText(frame, "FPS: " + str(round(frps,2)), (10,275), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

        out.write(frame)

        # show the output frame
        cv2.imshow("Frame", frame)
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF

        fps.update()

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


    fps.stop()
    print("[INFO] Pre-process time: {:.2f}".format(pre_process))
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    import pandas as pd
    data = {
        'Model': [model_name],
        'Pre-Process Time': [pre_process],
        'Elapsed Time': [fps.elapsed()],
        'Avg. FPS': [fps.fps()]
    }

    df = pd.DataFrame(data)
    if flag == False:
        df.to_csv('Test_Data.csv', mode='a', index=False, header={'Model','Pre-Process Time','Elapsed Time','Avg. FPS'})
    else:
        df.to_csv('Test_Data.csv', mode='a', index=False, header=False)

    # do a bit of cleanup
    #vs.stop()
    vs.release()
    out.release()
    cv2.destroyAllWindows()

detect_video('saved_model/ResnetV2','saved_videos/ResnetV2_600.avi',False,'ResnetV2')
detect_video('saved_model/MobileNETV2','saved_videos/MNETV2.avi',False,'MobileNetV2')
