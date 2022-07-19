def main():
    import os
    import torch
    import numpy as np
    from torch.nn import CrossEntropyLoss
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from recognition.segment_based import model
    from recognition.segment_based import character_dataset

    #################################################################
    batch_size = 8
    char_path = 'dataset/characters/'
    lr = 0.001
    #################################################################
    model = model.Segment_character(36)
    dataset = character_dataset.Character_dataset(ori_path=char_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(len(dataloader))

    #################################################################
    CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    #################################################################
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #################################################################

    for epoch in range(5):
        losses = 0
        for i, (input, target) in enumerate(dataloader):
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # gradient clipping with 5
            optimizer.step()
            losses += loss
            #################################################

        print(f"EPOCH {epoch}, loss {losses}")
        print()
        print()


if __name__ == '__main__':
    main()
