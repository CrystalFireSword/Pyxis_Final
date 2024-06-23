import cv2
from ultralytics import YOLO
import torch
import time

# model
model = YOLO('yolov8n.pt')

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# counter

#capture = cv2.VideoCapture('Trafficvideo.mp4')
# capture = cv2.VideoCapture(0)


def detect():
    totalresults = []
    capture = cv2.VideoCapture(0)
    while True:
        is_frame, frame = capture.read()
        if not is_frame:
            break
        if torch.cuda.is_available():
            results = model.predict(frame, device = '0',  verbose = False, show = False)
                    
        else:
            results = model(frame, show = True)
        
        for result in results:
            totalresults.append(result.verbose())
            bboxes = result.boxes
            for bbox in bboxes:
                (x,y,x2,y2) = bbox.xyxy[0]
                x,y,x2,y2 = map(int,(x,y,x2,y2))
                cls = int(bbox.cls[0])
                cv2.rectangle(frame, (x,y), (x2,y2), (0,255,0,),2)
                org = [x, y]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness) 
                cv2.imshow('Video', frame)
        key = cv2.waitKey(1)  # to retain the image on the screen
        
        #if key == 27 or key == 13:
        if len(totalresults) >5:
            break
    
    capture.release()
    cv2.destroyAllWindows()

    print('\n Total Results \n')
    ret_res = []
    print(totalresults)
    for x,y in enumerate(totalresults):
        ret_res.append(f'Frame {x}: {y}'+'\n\n')
    totalresults = []
    return ret_res

    
print('yolo file done')
# wordnet - english
# similarity score 
# QA based LLMs
# LLM based chatbot with RAG - augmented generation
# kafka streaming
# bigdata streaming concepts
# vector store
# prompting 
# question to querying
# langchain framework
# llama model

