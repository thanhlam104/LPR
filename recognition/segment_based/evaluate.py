import os
import warnings
import torch

from utils import label2char, plot_img
from recognition.character_based.plate_dataset import Plate_Dataset
import numpy as np
import cv2
from torchmetrics import CharErrorRate
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')








def show_results(chars):
    from model import Segment_character
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    #################################################################
    model = Segment_character(36).to(device)
    checkpoint_path = 'checkpoint/character_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)

    #################################################################
    CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHAR2LABEL = {char: i for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    pred = []
    for i, ch in enumerate(chars):
        img = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = img.reshape(1, 1, 28, 28)
        img = torch.FloatTensor(img).cuda()
        output = model(img)[0].argmax()
        character = LABEL2CHAR[int(output)]
        pred.append(character)

    plate_number = ''.join(pred)

    return plate_number


def main():
    from segment_character import segment_characters
    cer = CharErrorRate()
    CER = []
    data_split = 'data_split/test.txt'
    dataset = Plate_Dataset(data_split, mode='test', tensor_format=False)
    for i, (img, box, target, plate_length) in enumerate(dataset):
        img = np.array(img, dtype='uint8')
        chars = segment_characters(img)
        pred_str = [show_results(chars)]
        target_str = label2char([target])

        cer_score = cer(pred_str, target_str)

        print(pred_str[0], target_str[0], cer_score)

        CER.append(cer_score)


    print(sum(CER) / len(CER))


if __name__ == '__main__':
    main()