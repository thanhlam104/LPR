import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Segment_character
from character_dataset import Character_dataset
import warnings
from torchmetrics import F1Score

warnings.filterwarnings("ignore")


#################################################################
def train_batch(model, train_data, optimizer, criterion, device):
    model.train()
    input, target = [d.to(device) for d in train_data]
    batch_size = input.size(0)

    output = model(input)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # gradient clipping with 5
    optimizer.step()
    return loss.item()


def evaluate(model, dataloader, criterion, device):
    f1 = F1Score(num_classes=36).to(device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    tot_count = 0
    tot_loss = 0
    F1 = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input, target = [d.to(device) for d in data]
            batch_size = input.size(0)

            output = model(input)
            loss = criterion(output, target)

            tot_count += batch_size
            tot_loss += loss.item()
            pred = output.argmax(dim=1)
            f1_score = f1(pred, target)
            F1.append(f1_score)

    evaluation = {
        "loss": tot_loss / tot_count,
        "f1": sum(F1) / len(F1)
    }

    return evaluation


def main():
    ##################--CONFIG--#########################
    batch_size = 32
    lr = 0.001
    epochs = 50
    show_interval = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_model = False
    save_model = True
    #####################################################
    char_path = 'dataset/characters/'
    checkpoint_path = 'checkpoint/character_model.pt'

    train_dataset = Character_dataset(ori_path=char_path, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = Character_dataset(ori_path=char_path, train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = Segment_character(36).to(device)
    if load_model:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint)

    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #####################################################
    i = 1

    best_f1 = 0
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0
        tot_train_count = 0

        for i, train_data in enumerate(train_loader):
            loss = train_batch(model, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size

            if i % show_interval == 0:
                print(f'train_batch_loss[{i}]: ', loss / train_size)

                evaluation = evaluate(model, val_loader, criterion, device)
                print('val_evaluation: loss = {loss}, f1 = {f1}'.format(**evaluation))

                if evaluation['f1'] > best_f1:
                    best_f1 = evaluation['f1']
                    save_model_path = f'checkpoint/character_model.pt'
                    torch.save(model.state_dict(), save_model_path)
                    print('=======> save model at ', save_model_path)
            i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)
        print()
        print()


if __name__ == '__main__':
    main()
