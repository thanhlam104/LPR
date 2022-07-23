import cv2
import torch
import numpy as np
from numpy import asarray
from detection.handcrafted import DetectChars, DetectPlates, PossiblePlate, PossibleChar, Preprocess
from dataset import Dataset
from utils import plot_img, box_center2corner, intersection_over_union, iou_per_box
import timeit

t0 = timeit.default_timer()

#################################################
def remove_overlapping_plate(list_possible_plate1, list_possible_plate2):
    list_possible_plate = []
    size1 = len(list_possible_plate1)
    if size1 == 1:
        return list_possible_plate1
    else:
        size2 = len(list_possible_plate2)
        if size1 > 0 and size2 > 0:
            corr = np.zeros((size1, size2))
            for i in range(size1):
                for j in range(size2):
                    box_pr_corner1 = box_center2corner(list_possible_plate1[i].box)
                    box_pr_corner2 = box_center2corner(list_possible_plate2[j].box)
                    iou = iou_per_box(torch.tensor(box_pr_corner1), torch.tensor(box_pr_corner2))
                    corr[i, j] = iou
                    if corr[i, j] > 0.6:
                        list_possible_plate.append(list_possible_plate1[i])
                        break

            if (corr < 0.6).all():
                idx = corr.argmax()
                list_possible_plate.append(list_possible_plate1[idx//size2])

        if len(list_possible_plate) > 3:
            list_possible_plate = [list_possible_plate1[0]]

    return list_possible_plate


def detect_box(data_split, mode='train'):
    dataset = Dataset(data_split)

    count_empty_detect = 0
    iou_dataset = []
    nonDetect_img = 0

    for i, (img, box_gt) in enumerate(dataset):
        list_possible_plate1 = DetectPlates.detect_plates(img, 19, 9)
        list_possible_plate2 = DetectPlates.detect_plates(img, 39, 1)
        list_possible_plate = remove_overlapping_plate(list_possible_plate1, list_possible_plate2)

        if len(list_possible_plate) > 0:
            iou_plate = []

            for j, plate in enumerate(list_possible_plate):
                box_pr_corner = box_center2corner(plate.box)
                iou = iou_per_box(torch.tensor(box_pr_corner), torch.tensor(box_gt))
                iou_plate.append(iou)
            iou_dataset.append(iou_plate[0])
            cv2.imwrite(f'dataset/plates/{mode}/img{i}_{j}.jpg', list_possible_plate[j].img_plate)

        else:
            count_empty_detect += 1
            iou_dataset.append(0)
            nonDetect_img += 0
            cv2.imwrite(f'dataset/plates/{mode}/img{i}_{j}.jpg', img)

        if i % 100 == 0:
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
    time = t1 - t0

    return iou_dataset, nonDetect_img, zero_iou, pc, time


if __name__ == '__main__':
    ious, nonDetect_img, zero_iou, pc, time = detect_box('data_split/test.txt', mode='hmm')
    print("total image", len(ious))
    print("non detect image", nonDetect_img)
    print("zero iou", zero_iou)
    print("mean iou", sum(ious) / len(ious))
    print("precision", pc / len(ious))
    print("detect time", time)