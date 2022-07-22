from torch.nn import CTCLoss
from recognition.character_based import model_chacter_based
import warnings
import torch
from torchmetrics import CharErrorRate
from plate_dataset import Plate_Dataset
from torch.utils.data import DataLoader
from ctc_decoder import ctc_decode
from utils import label2char

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#################################################################


batch_size = 32

cer = CharErrorRate()


def evaluate(crnn, dataloader, criterion, decode_method='beam_search', beam_size=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    crnn.eval()

    tot_count = 0
    tot_loss = 0
    CER = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input, box_gt, target, plate_length = [d.to(device) for d in data]

            logits = crnn(input)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = input.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
            plate_length = torch.tensor(plate_length).to(device)

            loss = criterion(log_probs, target, input_lengths, plate_length)
            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)

            tot_count += batch_size
            tot_loss += loss.item()

            preds_str = label2char(preds)
            target_str = label2char(target)


            cer_score = cer(preds_str, target_str)
            CER.append(cer_score)

    evaluation = {
        "loss": tot_loss / tot_count,
        "cer": sum(CER) / len(CER)
    }

    return evaluation


if __name__ == "__main__":
    model = model_chacter_based.CRNN(1, 32, 128, 37).to(device)
    checkpoint_path = 'checkpoint/crnn.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    criterion = CTCLoss(reduction='sum').to(device)

    train_dataset = Plate_Dataset('data_split/train.txt')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = Plate_Dataset('data_split/val.txt', mode='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataset = Plate_Dataset('data_split/test.txt', mode='test')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)

    train_evaluation = evaluate(model, val_loader, criterion, decode_method='beam_search', beam_size=10)
    val_evaluation = evaluate(model, val_loader, criterion, decode_method='beam_search', beam_size=10)
    test_evaluation = evaluate(model, test_loader, criterion, decode_method='beam_search', beam_size=10)

    print('train cer', train_evaluation['cer'])
    print('val cer', val_evaluation['cer'])
    print('test cer', test_evaluation['cer'])
