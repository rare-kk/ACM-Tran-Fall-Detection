
''' YOLO V2 MODEL FALL DETECTION '''

import numpy as np
import cv2
import time
from math import sqrt 

labelsPath = "D://BHU_MSc//MSc_SEM_4//YOLO//coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath="D:\BHU_MSc\MSc_SEM_4\YOLO\yolov2.weights"
configPath = "D:\BHU_MSc\MSc_SEM_4\YOLO\yolov2.cfg"

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames() 
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture("E:/test.mp4")
(W, H) = (None, None)

# Accuracy and detection 
f_thresh= 0.9
ratio = 0 
cnt = 0 
fno = 0
conf = 0.6
threshold = 0.6
l = 1
defaultsize= tuple((700,400))
counter = 0
detect=0
tp=0
tn=0

start= time.time()
while True:
    (ret, frame) = vs.read()
    if not ret:
        break
    fno+=1
    #if cnt%2==0:
     #   cnt+=1
      #  continue
  #cv2.imshow("check",frame)
    frame = cv2.resize(frame, defaultsize)
    print("Frame No:", fno)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob) 
    layerOutputs = net.forward(ln) # all layers of the frame 

    boxes = []
    confidences = []
    classIDs = []
    bboxes = []
    confidences = []
    classIDs = []
    means1 = []
    means2 = []
    means3 = []
    loss = []
    coord=5
    noord=0.5
        
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sq(a,b):
        c = pow((a-b),2)
        return c
        
    def sqroot(a, b):
        d = pow((sqrt(a) - sqrt(b)), 2)
        return d

    #loop over each of the layer outputs
    for output in layerOutputs:
    # loop over each of the detections of each layer
        for detection in output: 
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores) #return highest value index from array of object
            confidence = scores[classID]

            # For Loss Function
            # predicted value of bounding box coordinate x,y,w,h and confidence 
            # of box  and the class probability
            pred_x, pred_y, pred_w, pred_h , pred_p, pred_c  = detection[0:6]
            
            #ground truth value calculated from sigmoid function. 
            gt_x = sigmoid(detection[0])
            gt_y = sigmoid(detection[1])
            gt_w = sigmoid(detection[2])
            gt_h = sigmoid(detection[3])
            gt_p = sigmoid(detection[4])
            gt_c = sigmoid(detection[5])
            
            # square and square root for loss function 
            
            lx = sq(pred_x, gt_x)
            ly = sq(pred_y, gt_y)
            lw = sqroot(pred_w, gt_w)
            lh = sqroot(pred_h, gt_h)
            
            #Loss of bounding box coordinate 
            mean1 = ((lx+ly+lw+lh)*coord)
            means1.append(mean1)
            #print("means1: ", means1)
            
            # Loss of class of object detected 
            lc = sq(pred_c, gt_c)
            means3.append(lc)
            
            #Loss of confidence of bounding boxes
            if (confidence > conf):
                
                lp1 = sq(pred_p, gt_p)*coord
                means2.append(lp1)
            else:
                lp2 = sq(pred_p, gt_p)*noord    
                means2.append(lp2)    

      # filter out weak predictions by ensuring the detected
      # probability is greater than the minimum probability
            if confidence > conf :
                detectt= detection[0:4]
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs

                bboxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
             
        k= len(means1)
        #loss_ind = (a+b+c)/k
        loss_ind = (sum(means1) + sum(means2) + sum(means3))/k
        #print("loss:", p) 
        loss.append(loss_ind)
        avgloss = sum(loss)
                    
  # apply non-maxima suppression to suppress weak, overlapping
  # bounding boxes
    idxs = cv2.dnn.NMSBoxes(bboxes, confidences, conf, threshold)

  # ensure at least one detection exists
    if len(idxs) > 0:
        for i in idxs.flatten():
           # extract the bounding box coordinates
           detect+=1
           if(LABELS[classIDs[i]] == "person"):
                (x, y) = (bboxes[i][0], bboxes[i][1])
                (w, h) = (bboxes[i][2], bboxes[i][3])
                if(h > 0 )and (w > 0):
                  ratio= h / w
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                if(ratio>f_thresh):
                    id = "ratio: {:.4f}\n NON-FALL".format(ratio)
                    tn+=1
                else:
                    id = "ratio: {:.4f} \n FALL".format(ratio)
                    tp+=1
                
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                cv2.putText(frame, id, (x+3, y+13),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    cnt+=1
    cv2.imshow('video',frame)
    Accuracy= (tp+tn)/fno
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
end = time.time()
print('Accuracy of YOLO v2 Model : ', Accuracy)
print('Execution time in second : ', end-start)
print('Loss Function of YOLO v2 : ', avgloss)




