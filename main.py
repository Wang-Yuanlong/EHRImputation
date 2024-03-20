import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
from tqdm import tqdm
# from dataset import EHRData, lab_event_list
from dataset import ImpData, imp_event_list
from model import Imputer
from metric import compute_nRMSE
# from sklearn.metrics import roc_auc_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=30)
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-t', '--test_only', action='store_true')
parser.add_argument('--bidirectional', action='store_true')

args = parser.parse_args()
print(args)
epochs = args.epochs
batch_size = args.batch_size
test_only = args.test_only
bidirectional = args.bidirectional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset = EHRData()
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=EHRData.get_collate())
# test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=EHRData.get_collate())

dataset = ImpData()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=ImpData.get_collate())
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=ImpData.get_collate())

model = Imputer(bidirectional=bidirectional).to(device)
criterion = torch.nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    item_num=0
    # for idx, (data, missing_mask, target_mask) in enumerate((train_loader)):
        # data, missing_mask, target_mask = data.to(device), missing_mask.to(device), target_mask.to(device)
        # input_ts = data * (1 - target_mask)
        # output = model(input_ts, missing_mask)
    for idx, (data, data_mask, target_mask, gt) in enumerate((train_loader)):
        data, data_mask, target_mask, gt = data.to(device), data_mask.to(device), target_mask.to(device), gt.to(device)
        output = model(data, data_mask)
        # loss = criterion(output * target_mask, data * target_mask)
        loss = criterion(output, gt)
        valid_loss = loss[data_mask == 1]
        loss_m = valid_loss.mean()
        
        loss_m.backward()
        optimizer.zero_grad()
        optimizer.step()
        total_loss += valid_loss.sum().item()
        item_num += len(valid_loss)

        if idx % 50 == 0:
            print(f'Batch [{idx+1}/{len(train_loader)}] Loss: {loss_m.item():.4f}')
    return total_loss / item_num

@torch.no_grad()
def val_epoch(model, test_loader, criterion, device):
    model.eval()
    total_init_loss = 0
    total_target_loss = 0
    total_loss = 0
    init_item_num = 0
    target_item_num = 0
    item_num=0
    all_pred = []
    all_gt = []
    all_mask = []
    # for idx, (data, missing_mask, target_mask) in enumerate(tqdm(test_loader)):
    #     data, missing_mask, target_mask = data.to(device), missing_mask.to(device), target_mask.to(device)
    #     input_ts = data * (1 - target_mask)
    #     output = model(input_ts, missing_mask)
    for idx, (data, data_mask, target_mask, gt) in enumerate((test_loader)):
        data, data_mask, target_mask, gt = data.to(device), data_mask.to(device), target_mask.to(device), gt.to(device)
        output = model(data, data_mask)
        # loss = criterion(output * target_mask, data * target_mask)
        loss = criterion(output, gt)
        init_mask = data_mask - target_mask

        init_loss = loss[init_mask == 1]
        target_loss = loss[target_mask == 1]
        valid_loss = loss[data_mask == 1]

        mask = data_mask - 1 + target_mask
        all_pred += [x for x in output]
        all_gt += [x for x in gt]
        all_mask += [x for x in mask]
        total_loss += loss.item()
        item_num += len(valid_loss)
        total_init_loss += init_loss.sum().item()
        init_item_num += len(init_loss)
        total_target_loss += target_loss.sum().item()
        target_item_num += len(target_loss)
    all_pred = pad_sequence(all_pred, batch_first=True, padding_value=0)
    all_gt = pad_sequence(all_gt, batch_first=True, padding_value=0)
    all_mask = pad_sequence(all_mask, batch_first=True, padding_value=-1)
    metric_list = compute_nRMSE(all_pred.cpu().numpy(), all_gt.cpu().numpy(), all_mask.cpu().numpy())
    return (total_loss / item_num, total_init_loss / init_item_num, total_target_loss / target_item_num), np.mean(metric_list[:2])

@torch.no_grad()
def test_epoch(model, test_loader, criterion, device):
    model.eval()
    all_pred = []
    all_gt = []
    all_mask = []
    # for idx, (data, missing_mask, target_mask) in enumerate((test_loader)):
    #     data, missing_mask, target_mask = data.to(device), missing_mask.to(device), target_mask.to(device)
    #     input_ts = data * (1 - target_mask)
    #     output = model(input_ts, missing_mask)
    for idx, (data, data_mask, target_mask, gt) in enumerate((test_loader)):
        data, data_mask, target_mask, gt = data.to(device), data_mask.to(device), target_mask.to(device), gt.to(device)
        output = model(data, data_mask)
        mask = data_mask - 1 + target_mask
        all_pred += [x for x in output]
        all_gt += [x for x in gt]
        all_mask += [x for x in mask]
    # all_loss: (batch_size, seq_len, variable_num)
    # all_loss = list(map(lambda x: x.cpu().reshape(-1, x.shape[-1]), all_loss))
    # all_mask = list(map(lambda x: x.cpu().reshape(-1, x.shape[-1]), all_mask))
    all_pred = pad_sequence(all_pred, batch_first=True, padding_value=0)
    all_gt = pad_sequence(all_gt, batch_first=True, padding_value=0)
    all_mask = pad_sequence(all_mask, batch_first=True, padding_value=-1)
    metric_list = compute_nRMSE(all_pred.cpu().numpy(), all_gt.cpu().numpy(), all_mask.cpu().numpy())
    return metric_list

@torch.no_grad()
def best_test(model, test_loader, criterion, device):
    model.load_state_dict(torch.load('saved_model/best_model.pth'))
    model.eval()
    # print('\t' + ','.join(lab_event_list) + ',mean,std,support')
    print('\t' + ','.join(imp_event_list))
    dataset.set_split('train')
    losses = test_epoch(model, test_loader, criterion, device)
    # print('train\t' + ','.join([f'{x:.4f}' for x in losses] + [f'{x:.4f}' for x in (losses.mean(), losses.std(), len(dataset))]))
    print('train\t' + f'{losses}')
    dataset.set_split('val')
    losses = test_epoch(model, test_loader, criterion, device)
    # print('val\t' + ','.join([f'{x:.4f}' for x in losses] + [f'{x:.4f}' for x in (losses.mean(), losses.std(), len(dataset))]))
    print('val\t' + f'{losses}')
    dataset.set_split('test')
    losses = test_epoch(model, test_loader, criterion, device)
    # print('test\t' + ','.join([f'{x:.4f}' for x in losses] + [f'{x:.4f}' for x in (losses.mean(), losses.std(), len(dataset))]))
    print('test\t' + f'{losses}')

def train(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    best_RMSE = float('inf')
    for epoch in range(epochs):
        dataset.set_split('train')
        print(f'\nEpoch [{epoch+1}/{epochs}]')
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_loss_list, train_RMSE = val_epoch(model, test_loader, criterion, device)
        dataset.set_split('val')
        val_loss, val_RMSE = val_epoch(model, test_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Train RMSE: {train_RMSE:.4f} Val RMSE: {val_RMSE:.4f}')
        print('Breakdown loss - Train: ({:.4f},{:.4f},{:.4f})'.format(*train_loss_list))
        print('Breakdown loss - Val: ({:.4f},{:.4f},{:.4f})'.format(*val_loss))
        if val_RMSE < best_RMSE:
            print(f'New best model with RMSE: {best_RMSE:.4f} -> {val_RMSE:.4f}')
            best_RMSE = val_RMSE
            torch.save(model.state_dict(), 'saved_model/best_model.pth')
    dataset.set_split('test')
    test_loss, test_RMSE = val_epoch(model, test_loader, criterion, device)
    print(f'Test Loss: ({test_loss[0]:.4f}, {test_loss[1]:.4f}, {test_loss[2]:.4f}) Test RMSE: {test_RMSE:.4f}')
    best_test(model, test_loader, torch.nn.MSELoss(reduction='none'), device)

if __name__ == '__main__':
    if test_only:
        best_test(model, test_loader, torch.nn.MSELoss(reduction='none'), device)
    else:
        train(model, train_loader, test_loader, criterion, optimizer, device, epochs)
    

    

