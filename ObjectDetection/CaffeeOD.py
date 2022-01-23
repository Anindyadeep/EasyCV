import os
import cv2
import numpy as np
import FaceRecognition.recognize_face as rf

class CamObjectDetection(object):
    def __init__(self, cam_num : int, class_names : dict, model_config : dict, threshold : float = 0.2):
        '''
        This will perform the real time object detection on the camera
        Parameters:
        -----------
        cam_num : The camera number for initializing the Object detection,
        class_names : All the class names required for recognition
        threshold : the minimum thershold for detection (default : 0.2)
        '''

        self.cam_num = cam_num
        self.class_names = class_names
        self.threshold = threshold
        self.model_config = model_config

        __base_path = os.getcwd()[:-5]
        self.rf = rf.FaceRecognition(face_encoding_path=os.path.join(__base_path, 'registered_face_encodings'))
    
    def load_net_from_caffee(self):
        '''
        Will take the .prototxt and .weights path to build the model
        '''

        prototxt_path = self.model_config['prototxt_path']
        weights_path  = self.model_config['weights_path']
        net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
        return net 

    def __get_color(self):
        np.random.seed(1)
        colors = np.random.uniform(0, 255, size=(len(self.class_names.values()),3))
        return colors
    
    def perform_object_detection(self, config, show_logs = False):
        if config is None:
            config = {
                'resize_frame' : True,
                'image_input'  : (300, 300)
            }
        
        face_locations = []
        face_encodings = []
        face_names = []

        net = self.load_net_from_caffee()
        resize_frame = True if config['resize_frame'] is None else config['resize_frame']
        image_input = (300, 300) if config['image_input'] is None else config['image_input']
        box_colors = self.__get_color()
        cap = cv2.VideoCapture(self.cam_num)
        while True:
            ret, frame = cap.read()
            if resize_frame:
                frame_resized = cv2.resize(frame, image_input)
            else:
                frame_resized = frame 
            blob = cv2.dnn.blobFromImage(frame_resized, 0.00783, image_input, (127.5, 127.5, 127.5), False)
            net.setInput(blob)
            detections = net.forward()

            cols, rows = frame_resized.shape[0], frame_resized.shape[1] 
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.threshold:
                    class_id = int(detections[0, 0, i, 1])

                    xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                    yLeftBottom = int(detections[0, 0, i, 4] * rows)
                    xRightTop   = int(detections[0, 0, i, 5] * cols)
                    yRightTop   = int(detections[0, 0, i, 6] * rows)

                    label = self.class_names[class_id]
                    if label == "person":
                        length = np.abs((xLeftBottom - xRightTop))
                        width = np.abs((yLeftBottom - yRightTop))
                        area = length * width 

                        if area > 26000:
                            frame, face_locations, face_encodings, face_names = self.rf.recognize_face_per_frame(
                            frame,
                            face_locations,
                            face_encodings,
                            face_names
                        )
                        else:
                            print("Face detection model shuted down ...") 
                        if show_logs:
                            print(f"Detected Person, Area: {area} units")
                    
                    if resize_frame:
                        heightFactor = frame.shape[0]/300.0  
                        widthFactor = frame.shape[1]/300.0 

                        xLeftBottom = int(widthFactor * xLeftBottom) 
                        yLeftBottom = int(heightFactor * yLeftBottom)
                        xRightTop   = int(widthFactor * xRightTop)
                        yRightTop   = int(heightFactor * yRightTop)
                    
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          box_colors[class_id])
                    
                    
                    if class_id in self.class_names:
                        label = self.class_names[class_id] + ": " + str(confidence)
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        yLeftBottom = max(yLeftBottom, labelSize[1])
                        cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]), (xLeftBottom + labelSize[0], yLeftBottom + baseLine), (255, 255, 255), cv2.FILLED)
                        cv2.putText(frame, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                        

            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) >= 0 or (cv2.waitKey(25) & 0xFF == ord("q")):
                break 
        cv2.destroyAllWindows()


if __name__ == '__main__':
    base_path = os.getcwd()[:-5]
    model_config = {
        'prototxt_path' : os.path.join(base_path, 'Model/MobileNetSSD_deploy.prototxt'),
        'weights_path'  : os.path.join(base_path, 'Model/MobileNetSSD_deploy.caffemodel')
    }
    class_names = { 0: 'background',
                    1: 'aeroplane', 
                    2: 'bicycle', 
                    3: 'bird', 
                    4: 'boat',
                    5: 'bottle', 
                    6: 'bus', 
                    7: 'car', 
                    8: 'cat', 
                    9: 'chair',
                    10: 'cow', 
                    11: 'diningtable', 
                    12: 'dog', 
                    13: 'horse',
                    14: 'motorbike', 
                    15: 'person', 
                    16: 'pottedplant',
                    17: 'sheep', 
                    18: 'sofa', 
                    19: 'train', 
                    20: 'tvmonitor' }

    cod = CamObjectDetection(cam_num=0, class_names=class_names, model_config=model_config)
    cod.perform_object_detection(config=None, show_logs=True)
