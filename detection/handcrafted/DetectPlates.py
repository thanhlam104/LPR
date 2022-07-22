import os
import cv2
import math
import random
import numpy as np
import torch

from detection.handcrafted import Preprocess
from detection.handcrafted import PossibleChar
from detection.handcrafted import PossiblePlate
from detection.handcrafted import DetectChars
from utils import box_center2corner, iou_per_box

# module level variables
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


##################################
def detect_plates(img_original, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT):
    list_possible_plates = []

    h, w, c = img_original.shape

    img_gray_scale = np.zeros((h, w, 1), np.uint8)
    img_thresh = np.zeros((h, w, 1), np.uint8)
    img_contours = np.zeros((h, w, 1), np.uint8)

    img_gray_scale, img_thresh = Preprocess.preprocess(img_original, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    list_possible_chars = find_possible_chars(img_thresh)

    list_of_list_matching_chars = DetectChars.find_list_of_list_matching_chars(list_possible_chars)

    for list_matching_chars in list_of_list_matching_chars:
        possible_plate = extract_plate(img_original, list_matching_chars)

        if possible_plate.img_plate is not None:
            list_possible_plates.append(possible_plate)

    return list_possible_plates


def find_possible_chars(img_thesh):
    list_possible_chars = []
    count_possible_chars = 0
    img_thresh_copy = img_thesh.copy()

    contours, hierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img_thesh.shape
    img_contours = np.zeros((h, w, 3), np.uint8)  # for visual

    for i in range(0, len(contours)):
        possible_char = PossibleChar.PossibleChar(contours[i])

        if DetectChars.check_possible_char(possible_char):
            count_possible_chars += 1
            list_possible_chars.append(possible_char)

    return list_possible_chars


def extract_plate(img_original, list_matching_chars):
    possible_plate = PossiblePlate.PossiblePlate()  # return value

    list_matching_chars.sort(key=lambda matching_char: matching_char.box_x)  # sort char from left to right

    plate_centerX = (list_matching_chars[0].box_centerX + list_matching_chars[-1].box_centerX) / 2
    plate_centerY = (list_matching_chars[0].box_centerY + list_matching_chars[-1].box_centerY) / 2

    plate_center = plate_centerX, plate_centerY

    # calculate width and height
    plate_w = int((list_matching_chars[-1].box_x + list_matching_chars[-1].box_w - list_matching_chars[
        0].box_x) * PLATE_WIDTH_PADDING_FACTOR)

    total_char_heights = 0

    for matching_chars in list_matching_chars:
        total_char_heights += matching_chars.box_h

    avg_char_heights = total_char_heights / len(list_matching_chars)

    plate_h = int(avg_char_heights * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    opposite = list_matching_chars[-1].box_centerY - list_matching_chars[0].box_centerY
    hypotenuse = DetectChars.distance_between_chars(list_matching_chars[0], list_matching_chars[-1])
    correction_angle_rad = math.asin(opposite / hypotenuse)
    correction_angle_deg = correction_angle_rad * 180 / math.pi

    # pack plate region center point, width, height, correction angle
    possible_plate.rrLocationOfPlateInScene = (tuple(plate_center), (plate_w, plate_h), correction_angle_deg)

    # box information x, y, h, w format
    possible_plate.box = [plate_centerX, plate_centerY, plate_h, plate_w]

    # get the rotation  matrix for correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(plate_center), correction_angle_deg, 1.0)

    h, w, c = img_original.shape

    img_rotated = cv2.warpAffine(img_original, rotationMatrix, (w, h))  # rotate the entire image

    img_cropped = cv2.getRectSubPix(img_rotated, (plate_w, plate_h), tuple(plate_center))
    possible_plate.img_plate = img_cropped

    return possible_plate



