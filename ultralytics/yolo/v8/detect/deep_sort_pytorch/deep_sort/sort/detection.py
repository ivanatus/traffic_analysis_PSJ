# vim: expandtab:ts=4:sw=4
import numpy as np
import sys
import os
linux_path = os.path.expanduser("~/traffic_analysis_PSJ/ultralytics/yolo/v8/detect/deep_sort_pytorch/deep_sort/sort")
sys.path.append(linux_path)

#sys.path.append(r"C:\Users\User\YOLOv8-DeepSORT-Object-Tracking\ultralytics\yolo\v8\detect\deep_sort_pytorch\deep_sort\sort")
#sys.path.append("~/YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect/deep_sort_pytorch/deep_sort/sort")

from globals import Globals
import csv
import datetime


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    global_instance = Globals()

    def __init__(self, tlwh, confidence, feature, oid):
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.oid = oid
        

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        
        #self.save_movement_to_csv('results_TEST.csv', Detection.global_instance.get_global_frame(), Detection.global_instance.get_no_of_people(), Detection.global_instance.get_no_of_bikes(), Detection.global_instance.get_no_of_buses(),Detection.global_instance.get_no_of_cars(), Detection.global_instance.get_no_of_trucks(), Detection.global_instance.get_no_of_trains())
        #self.save_movement_to_csv('result_TEST.csv')

        return ret

    
    def save_movement_to_csv(self, filename, frame, no_of_people, no_of_bikes, no_of_buses, no_of_cars, no_of_trucks, no_of_trains):
        """
        Save this frames data to csv file.
        """
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['frame', 'people', 'bikes', 'buses', 'cars', 'trucks', 'trains']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'frame': frame, 'people': no_of_people, 'bikes': no_of_bikes, 'buses': no_of_buses, 'cars': no_of_cars, 'trucks': no_of_trucks, 'trains': no_of_trains})
