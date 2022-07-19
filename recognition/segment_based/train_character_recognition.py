def main():
    import os
    import warnings
    import torch
    import numpy as np
    from torch.nn import CrossEntropyLoss, NLLLoss
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from recognition.segment_based import model
    from recognition.segment_based import character_dataset
    warnings.filterwarnings("ignore")
    #################################################################
    batch_size = 32
    char_path = 'dataset/characters/'
    lr = 0.001
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(device)
    #################################################################
    model = model.Segment_character(36).to(device)
    dataset = character_dataset.Character_dataset(ori_path=char_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset_val = character_dataset.Character_dataset(ori_path=char_path, train=False)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    print(len(dataloader_val), len(dataset_val))

    #################################################################
    CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    #################################################################
    criterion = CrossEntropyLoss()
    # criterion = NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #################################################################

    for epoch in range(30):
        losses = []
        accs = []
        for i, (input, target) in enumerate(dataloader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # gradient clipping with 5
            optimizer.step()
            losses.append(loss.item())

            acc = accuracy(output, target)
            accs.append(acc.item())
            #################################################

        losses_val = []
        accs_val = []
        for i, (input, target) in enumerate(dataloader_val):
            input, target = input.to(device), target.to(device)
            output = model(input)

            loss = criterion(output, target)
            acc = accuracy(output, target)
            losses_val.append(loss.item())
            accs_val.append(acc.item())


        print(f"EPOCH {epoch+1}, train loss {mean(losses)}, train accuracy {mean(accs)}, "
              f"val loss {mean(losses_val)}, val accuracy {mean(accs_val)}")
        print()

    #################################################################



def accuracy(output, target):
    pred = output.argmax(dim=1)
    acc = sum(pred == target) / len(pred)
    return acc

def mean(arr):
    m = round(sum(arr) / len(arr), 3)
    return m


if __name__ == '__main__':
    main()
