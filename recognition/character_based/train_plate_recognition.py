import torch
from torch.nn import CTCLoss
import torch.optim as optim
from torch.utils.data import DataLoader
from plate_dataset import Plate_Dataset
from recognition.character_based import model
from evaluate import evaluate
from utils import plot_img
import warnings

warnings.filterwarnings('ignore')


#################################################################
def train_batch(crnn, train_data, optimizer, criterion, device):
    crnn.train()
    input, box_gt, target, plate_length = [d.to(device) for d in train_data]

    logits = crnn(input)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = input.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    plate_length = torch.tensor(plate_length).to(device)

    loss = criterion(log_probs, target, input_lengths, plate_length)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5)  # gradient clipping with 5
    optimizer.step()
    return loss.item()


def main():
    ##################--CONFIG--#########################
    batch_size = 32
    lr = 0.0005
    epochs = 20

    show_interval = 47
    val_interval = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_model = True
    save_model = True

    #####################################################
    train_split = 'data_split/train.txt'
    val_split = 'data_split/val.txt'
    checkpoint_path = 'checkpoint/crnn.pt'


    train_dataset = Plate_Dataset(train_split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = Plate_Dataset(val_split, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    print("train loader:", len(train_loader), "val loader:", len(val_loader))

    crnn = model.CRNN(1, 32, 128, 37).to(device)

    if load_model:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        crnn.load_state_dict(checkpoint)

    criterion = CTCLoss(reduction='sum').to(device)
    optimizer = optim.Adam(crnn.parameters(), lr=lr)

    #####################################################
    i = 1
    best_cer = 0.266
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0
        tot_train_count = 0

        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print(f'train_batch_loss[{i}]: ', loss / train_size)

            if i % val_interval == 0:
                evaluation = evaluate(crnn, val_loader, criterion,
                                      decode_method='beam_search',
                                      beam_size=10)

                print('valid_evaluation: loss = {loss}, cer = {cer}'.format(**evaluation))

                if save_model and best_cer > evaluation['cer']:
                    best_cer = evaluation['cer']
                    save_model_path = f'checkpoint/crnn.pt'
                    # torch.save(crnn.state_dict(), save_model_path)
                    print('============> save model at ', save_model_path)
            i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)
        print("#####################################################\n\n")


if __name__ == '__main__':
    main()
