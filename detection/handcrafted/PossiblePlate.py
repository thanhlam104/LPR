import cv2
import numpy as np


class PossiblePlate:
    def __init__(self):
        self.img_plate = None
        self.img_gray = None
        self.img_thresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""

        # box: center x, center y, height, width
        self.box = None