import cv2
import torch
from tqdm import tqdm
import numpy as np
from numpy import asarray
import os
from detection.handcrafted import DetectChars, DetectPlates, PossiblePlate, PossibleChar, Preprocess
from dataset import Dataset
from utils import plot_img, box_center2corner, intersection_over_union, iou_per_box

data_split = 'data_split/train.txt'


dataset = Dataset(data_split)
boxes_gt = []
boxes_pr = []
count_empty_detect = 0
iou_dataset = []
hard_img = []


for i, (img, box_gt, plate_label, plate_length) in enumerate(dataset):
    list_possible_plate = DetectPlates.detect_plates(img)
    if len(list_possible_plate) > 0:
        iou_plate = []
        for plate in list_possible_plate:
            box_pr_corner = box_center2corner(plate.box)
            iou = iou_per_box(torch.tensor(box_pr_corner), torch.tensor(box_gt))
            iou_plate.append(iou)
        iou_dataset.append(max(iou_plate))
        cv2.imwrite(f'img_plate/img{i}.jpg', list_possible_plate[asarray(iou_plate).argmax()].img_plate)
    else:
        # box_pr_corner = [0, 0, 0, 0]
        count_empty_detect += 1
        iou_dataset.append(0)
        hard_img.append(img)
        cv2.imwrite(f'img_plate/img{i}.jpg', img)



iou_dataset = asarray(iou_dataset)