import cv2
import sys
import math
import glob
import time
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectDetector():
    def __init__(self, model_name):
        """
        Initializes an ObjectDetector instance.

        Args:
            model_name (str): The name of the model to be loaded.

        Returns:
            None
        """
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))

    def load_model(self, model_name):
        """
        Loads the specified model.

        Args:
            model_name (str): The name of the model to be loaded.

        Returns:
            model: The loaded model.
        """
        if model_name:
            model = YOLO(model_name)
        else:
            model = YOLO("yolov8n.pt")
        return model
    
    def score_frame(self, frame):
        """
        Scores the given frame using the loaded model.

        Args:
            frame (numpy.ndarray): The frame to be scored.

        Returns:
            tuple: A tuple containing the labels, coordinates, and confidence scores of the detected objects.
        """
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))

        results = self.model(frame)

        labels, cord, conf = results[0].boxes.cls.tolist(), results[0].boxes.xyxyn.tolist(), results[0].boxes.conf.tolist()

        return labels, cord, conf
    
    def class_to_label(self, x):
        """
        Converts the class index to its corresponding label.

        Args:
            x (int): The class index.

        Returns:
            str: The corresponding label.
        """
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame, height, width, confidence=0):
        """
        Plots bounding boxes on the given frame based on the results.

        Args:
            results (tuple): A tuple containing the labels, coordinates, and confidence scores of the detected objects.
            frame (numpy.ndarray): The frame to be plotted on.
            height (int): The height of the frame.
            width (int): The width of the frame.
            confidence (float, optional): The confidence threshold for displaying the bounding boxes. Defaults to 0.1.

        Returns:
            tuple: A tuple containing the frame with plotted bounding boxes and a list of detections.
        """
        labels, cord, conf = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]
            con = conf[i]

            if con >= confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                x_center = x1 + (x2-x1)
                y_center = y1 + (y2-y1)/2

                feature = self.class_to_label(labels[i])

                detections.append(([x1, y1, int(x2-x1), int(y2-y1)], con, feature))

        return frame, detections
    
cap = cv2.VideoCapture("los_angeles.mp4")

od = ObjectDetector(model_name="runs\\detect\\train\\weights\\best.pt")

tracker = DeepSort(max_age=5,
                   n_init=2,
                   nms_max_overlap=0.3,
                   nn_budget=None,
                   override_track_class=None,
                   embedder="mobilenet",
                   half=True,
                   bgr=True,
                   embedder_gpu=True,
                   embedder_model_name=None,
                   embedder_wts=None,
                   polygon=False,
                   today=None)

# initialize frame counter
frame_count = 0
center_point_prev_frame = []

tracking_objects = {}
tracking_id = 0

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break

    start = time.perf_counter()

    # Object detection block
    results = od.score_frame(img)
    img, detections = od.plot_boxes(results, img, img.shape[0], img.shape[1])

    # DeepSORT block
    deepsort_count = 0
    tracks = tracker.update_tracks(detections, frame=img)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        class_name = track.get_det_class()
        ltrb = track.to_tlbr()

        bbox = ltrb
        deepsort_count += 1

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 1)
        cv2.putText(img, f"{track_id}:{class_name}", (int(bbox[0]), int(bbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Custom tracking block
    center_points_curr_frame = []   # Center points of the bounding boxes
    for (bbox, conf, feature) in detections:
        x, y, w, h = bbox
        x_center = x + (w/2)
        y_center = y + (h/2)
        center_points_curr_frame.append((x_center, y_center))
        
        cv2.putText(img, f"{feature}", (x,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
    
    # only at the beginning we compare the current frame with the previous one
    frame_count += 1
    if frame_count <= 2:
        for pt in center_points_curr_frame:
            for pt2 in center_point_prev_frame:
                distance = math.hypot(pt[0]-pt2[0], pt[1]-pt2[1])
                if distance < 50:
                    tracking_objects[tracking_id] = pt
                    tracking_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_curr_frame_copy = center_points_curr_frame.copy()
        for id, pt2 in tracking_objects_copy.items():
            object_found = False
            for pt in center_points_curr_frame_copy:
                distance = math.hypot(pt[0]-pt2[0], pt[1]-pt2[1])
                # Update object position
                if distance < 50:
                    tracking_objects[id] = pt
                    object_found = True
                    if pt in center_points_curr_frame:
                        center_points_curr_frame.remove(pt)
                    continue
            # Remove the ID
            if not object_found:
                tracking_objects.pop(id)

        for pt in center_points_curr_frame:
            tracking_objects[tracking_id] = pt
            tracking_id += 1

    custom_count = len(tracking_objects)
    for id, pt in tracking_objects.items():
        cv2.putText(img, str(id), (int(pt[0]), int(pt[1]-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(img, (int(pt[0]),int(pt[1])), 5, (0, 255, 0), -1)
            
    # Make a copy of the points
    center_points_prev_frame = center_points_curr_frame.copy()

    end = time.perf_counter()
    fps = 1.0 / (end-start)

    cv2.putText(img, f"#Vehicles being tracked by DeepSORT: {deepsort_count}/{tracks[-1].track_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f"#Vehicles being tracked by Custom Tracker: {custom_count}/{tracking_id}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
