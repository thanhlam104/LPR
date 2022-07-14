def main():
    from recognition.character_based import ctc_decoder
    from recognition.character_based import model
    import torch
    import numpy as np
    from dataset import Dataset
    import os
    from PIL import Image
    from torch.nn import CTCLoss
    import torch.optim as optim

    #######################################
    model = model.CRNN(1, 32, 128, 37)
    checkpoint_path = 'checkpoint/crnn.pt'
    data_split = 'data_split/train.txt'
    plate_dir = 'img_plate/'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    #######################################
    CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    #################################################################
    criterion = CTCLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ###########################################################

    img_width, img_height = 128, 32
    dataset = Dataset(data_split)
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
            input = plates[i].unsqueeze(0)
            logits = model(input)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            input_lengths = torch.LongTensor([logits.size(0)])

            loss = criterion(log_probs, torch.tensor(plate_label), input_lengths, torch.tensor(plate_length))

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

if __name__ == '__main__':
    main()
