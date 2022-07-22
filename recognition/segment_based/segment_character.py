import cv2
from utils import plot_img
import numpy as np
import matplotlib.pyplot as plt
from detection.handcrafted import Preprocess


def find_contours(dimensions, img):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    x_cntr_list = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(
                intX)  # stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44, 24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(img, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

    # Return characters on ascending order with respect to the x-coordinate (most-left character first)

    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


def segment_characters(image):
    # Preprocess cropped license plate image

    img_lp = cv2.resize(image, (333, 75), interpolation=cv2.INTER_AREA)
    if len(img_lp.shape) == 3:
        img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    else:
        img_gray_lp = img_lp

    img_gray_lp = Preprocess.maximize_contrast(img_gray_lp)

    _, img_gray_lp = cv2.threshold(img_gray_lp, thresh=50, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img_binary_lp = cv2.erode(img_gray_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))


    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH / 6,
                  LP_WIDTH / 2,
                  LP_HEIGHT / 10,
                  2 * LP_HEIGHT / 3]


    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


if __name__ == '__main__':
    from evaluate import show_results
    for i in range(10):
        img = cv2.imread(f'dataset/plates/train/img{i}_0.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # plot_img(img)

        char = segment_characters(img)
        pred_str = [show_results(char)]
        # print(pred_str)

        for j in range(len(char)):
            plt.subplot(1, len(char) + 1, j + 1)
            plt.imshow(char[j], cmap='gray')
            plt.axis('off')
        plt.show()

