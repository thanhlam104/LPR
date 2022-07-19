import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from utils import plot_img


class Character_dataset(nn.Module):
    CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHAR2LABEL = {char: i for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, ori_path, img_shape=(28, 28), train=True):
        super(Character_dataset, self).__init__()
        self.ori_path = ori_path
        self.shape = img_shape
        self.train = train
        self.img_h, self.img_w = img_shape
        self.dataset = []
        self.labels = []
        self.load_data()

    def load_data(self):
        if self.train:
            path_ = os.path.join(self.ori_path, "train")
        else:
            path_ = os.path.join(self.ori_path, "val")

        classes = os.listdir(path_)
        for each_class in classes:
            label = self.CHAR2LABEL[each_class[-1]]
            each_class_path = os.path.join(path_, each_class)
            for img_name in os.listdir(each_class_path):
                img = Image.open(os.path.join(each_class_path, img_name)).convert('L')
                img = img.resize((self.img_w, self.img_h), resample=Image.BILINEAR)
                img = np.array(img)
                img = (img / 127.5) - 1.0
                img = torch.FloatTensor(img)

                self.dataset.append(img)
                self.labels.append(label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx], self.labels[idx]
        return img.unsqueeze(0), label


if __name__ == '__main__':
    root_path = 'dataset/characters'
    char_dataset = Character_dataset(root_path)
    for i, (img, label) in enumerate(char_dataset):
        if i % 30 == 0:
            print(img.shape, label)
            plot_img(img.permute(1, 2, 0))



