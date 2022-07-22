import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class Dataset(Dataset):
    CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, data_split, img_height=720, img_width=1080, tensor_format=False):
        self.data_split = data_split
        self.list_ = list(open(self.data_split, 'r'))
        self.tensor_format = tensor_format
        self.img_width = img_width
        self.img_height = img_height

    def __getitem__(self, idx):
        img_path = "./dataset" + self.list_[idx].strip()[1:]

        img = Image.open(img_path)

        img = np.array(img)


        if self.tensor_format:
            img = torch.FloatTensor(img.transpose(2, 0, 1))

        # label and box
        img_infor = open(img_path.replace('jpg', 'txt'))
        for line in img_infor:
            if 'plate' in line:
                plate = line.split()[-1]
                plate_label = [self.CHAR2LABEL[c] for c in plate]
                plate_length = [len(plate_label)]

            if 'corners' in line:
                box = line.split()[1:]
                x1, y1 = int(box[0].split(',')[0]), int(box[0].split(',')[1])
                x2, y2 = int(box[1].split(',')[0]), int(box[1].split(',')[1])
                x3, y3 = int(box[2].split(',')[0]), int(box[2].split(',')[1])
                x4, y4 = int(box[3].split(',')[0]), int(box[3].split(',')[1])
                # pascal_voc box
                x_min = min(x1, x4)
                x_max = max(x2, x3)
                y_min = min(y1, y2)
                y_max = max(y3, y4)
                box_corner = [x_min, y_min, x_max, y_max]

        return img, box_corner

    def __len__(self):
        return len(self.list_)


if __name__ == '__main__':
    data_split = 'data_split/train.txt'

    dataset = Dataset(data_split)
    for i, (img, box) in enumerate(dataset):
        print(img.shape)
        print(box)

        plt.imshow(img)
        plt.show()
        break
