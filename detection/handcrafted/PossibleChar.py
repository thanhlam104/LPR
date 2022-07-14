import cv2
import math


class PossibleChar:

    def __init__(self, _contour):
        self.contour = _contour

        self.boundingRect = cv2.boundingRect(self.contour)

        [x, y, w, h] = self.boundingRect

        self.box_x = x
        self.box_y = y
        self.box_w = w
        self.box_h = h

        self.box_area = self.box_w * self.box_h

        self.box_centerX = self.box_x + self.box_w // 2
        self.box_centerY = self.box_y + self.box_h // 2

        self.box_diagonal = math.sqrt((self.box_w ** 2) + (self.box_h ** 2))

        self.box_aspect_ratio = self.box_w / self.box_h
