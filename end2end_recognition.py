import cv2
import numpy as np
import torch.cuda
import matplotlib.pyplot as plt
from PIL import Image

from utils import plot_img, box_center2corner, intersection_over_union, iou_per_box, label2char
from detection.handcrafted import Preprocess, DetectChars, DetectPlates, Detect

import warnings
warnings.filterwarnings('ignore')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def end2end_character_based(img): # RGB image
    from recognition.character_based import model_chacter_based, ctc_decoder
    ####################---DETECTION---##########################
    list_possible_plate1 = DetectPlates.detect_plates(img, 19, 9)
    list_possible_plate2 = DetectPlates.detect_plates(img, 39, 1)
    list_possible_plate = Detect.remove_overlapping_plate(list_possible_plate1, list_possible_plate2)
    plates = []

    for plate in list_possible_plate:
        img_plate = plate.img_plate
        img_plate = cv2.resize(img_plate, (128, 32), interpolation =Image.BILINEAR)
        img_plate = np.array(img_plate)
        img_plate = cv2.cvtColor(img_plate, cv2.COLOR_RGB2GRAY)
        img_plate = (img_plate / 127.5) - 1.0
        img_plate = torch.FloatTensor(img_plate).unsqueeze(0)
        plates.append(img_plate)

    ####################---RECOGNITION---##########################
    crnn = model_chacter_based.CRNN(1, 32, 128, 37).to(device)
    checkpoint_path = 'checkpoint/crnn.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    crnn.load_state_dict(checkpoint)

    res = ''
    for plate_img in plates:
        logits = crnn(plate_img.unsqueeze(0).to(device))
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        preds = ctc_decoder.ctc_decode(log_probs, method='beam_search', beam_size=10)
        preds_str = label2char(preds)
        print("predict plate:", preds_str[0])

        res += preds_str[0] + ' \n '


    return res





def end2end_segment_based(img): # RGB image
    from recognition.segment_based import segment_character, evaluate, model
    model1 = model.Segment_character(36).to(device)
    checkpoint_path = 'checkpoint/character_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model1.load_state_dict(checkpoint)
    ####################---DETECTION---##########################
    list_possible_plate1 = DetectPlates.detect_plates(img, 19, 9)
    list_possible_plate2 = DetectPlates.detect_plates(img, 39, 1)
    list_possible_plate = Detect.remove_overlapping_plate(list_possible_plate1, list_possible_plate2)

    plates = []
    for plate in list_possible_plate:
        img_plate = plate.img_plate
        plates.append(img_plate)

    ####################---RECOGNITION---##########################
    res = ''
    for img in plates:
        img = np.array(img, dtype='uint8')
        chars = segment_character.segment_characters(img)
        # for j in range(len(chars)):
        #     plt.subplot(1, len(chars) + 1, j + 1)
        #     plt.imshow(chars[j], cmap='gray')
        #     plt.axis('off')
        # plt.show()

        pred_str = [evaluate.show_results(chars, model1)]
        print("predict plate:", pred_str[0])
        res += pred_str[0] + ' \n '

    return res