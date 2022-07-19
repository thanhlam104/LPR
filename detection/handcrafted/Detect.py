import cv2
import torch
from tqdm import tqdm
import numpy as np
from numpy import asarray
import os
from detection.handcrafted import DetectChars, DetectPlates, PossiblePlate, PossibleChar, Preprocess
from dataset import Dataset
from utils import plot_img, box_center2corner, intersection_over_union, iou_per_box

import timeit

t0 = timeit.default_timer()
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


        new_dir = f'dataset/plates/plate_9_19/img{i}/'
        os.mkdir(new_dir)
        for j, plate in enumerate(list_possible_plate):
            box_pr_corner = box_center2corner(plate.box)
            iou = iou_per_box(torch.tensor(box_pr_corner), torch.tensor(box_gt))
            iou_plate.append(iou)
            iou_dataset.append(max(iou_plate))
            cv2.imwrite(f'dataset/plates/plate_9_19/img{i}/img{i}_{j}.jpg', list_possible_plate[asarray(iou_plate).argmax()].img_plate)

    else:
        # box_pr_corner = [0, 0, 0, 0]
        count_empty_detect += 1
        iou_dataset.append(0)
        hard_img.append(img)

        new_dir = f'dataset/plates/plate_9_19/img{i}/'
        os.mkdir(new_dir)
        cv2.imwrite(f'dataset/plates/plate_9_19/img{i}/img{i}_0.jpg', img)

    if i % 500 == 0:
        print(i)


iou_dataset = asarray(iou_dataset)

zero_iou = 0
pc = 0
for iou in iou_dataset:
    if iou == 0:
        zero_iou += 1

    if iou > 0.5:
        pc += 1

t1 = timeit.default_timer()

print("hard img", len(hard_img))
print("zero iou", zero_iou)
print("mean iou", asarray(iou_dataset).mean())
print("precision", pc / 3000)
print("time", t1 - t0)
