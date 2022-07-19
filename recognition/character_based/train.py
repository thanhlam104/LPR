from recognition.character_based import ctc_decoder
from recognition.character_based import model
import torch
import numpy as np
from dataset import Dataset
import os
from PIL import Image
from torch.nn import CTCLoss
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#######################################
model = model.CRNN(1, 32, 128, 37).to(device)
checkpoint_path = 'checkpoint/crnn.pt'
data_split = 'data_split/train.txt'
plate_dir = 'dataset/plates/train'
checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
model.load_state_dict(checkpoint)

batch_size = 32
lr = 0.01
#######################################
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

#################################################################
criterion = CTCLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=lr)
###########################################################

img_width, img_height = 128, 32
dataset = Dataset(data_split)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(len(dataloader))
plate_names = os.listdir(plate_dir)
plates = []

##########################################
for name in plate_names:
    image = Image.open(os.path.join(plate_dir, name)).convert('L')
    image = image.resize((img_width, img_height), resample=Image.BILINEAR)
    image = np.array(image)
    image = (image / 127.5) - 1
    image = torch.FloatTensor(image)
    image = image.reshape(1, img_height, img_width)
    plates.append(image)

for epoch in range(11):
    losses = 0
    for i, (img, box_gt, plate_label, plate_length) in enumerate(dataset):
        input = plates[i].unsqueeze(0).to(device)
        target = torch.tensor(plate_label).to(device)


        logits = model(input)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        input_lengths = torch.LongTensor([logits.size(0)]).to(device)
        plate_length = torch.tensor(plate_length).to(device)

        loss = criterion(log_probs, target, input_lengths, plate_length)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # gradient clipping with 5
        optimizer.step()

        losses += loss

        ##################################################
        pred = ''
        gt = ''
        preds = ctc_decoder.ctc_decode(log_probs, method='beam_search', beam_size=10)
        for num in preds[0]:
            pred += LABEL2CHAR[num]
        for num in plate_label:
            gt += LABEL2CHAR[num]

        ##################################################


    print(f"EPOCH {epoch}, loss {losses}")
    print()
    print()
