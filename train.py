import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm
import os

from data import load_dataset
from model import RNN

import pdb

def train(model, optimizer, criterion, data, label, hidden):
    model.train()
    output, hidden = model(data, hidden)
#     pdb.set_trace()
    output = output.squeeze(1).squeeze(1)
    if output.isnan().sum().item() >=1:
        pdb.set_trace()
    loss = criterion(output, label)
    if loss.isnan().sum().item() >=1:
        pdb.set_trace()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
#     print(output)
    score = ((torch.sigmoid(output) >= 0.5).long() == label).sum().item()/data.shape[0]
    return score, loss.detach().cpu().numpy(), hidden

def test(model, criterion, data, label, hidden):
    model.eval()
    output, hidden = model(data, hidden)
    output = output.squeeze(1).squeeze(1)
    loss = criterion(output, label)
    
    correct = ((torch.sigmoid(output) >= 0.5).long() == label).sum().item()
#     pdb.set_trace()
    return correct, data.size(0), loss, hidden

def main():
    parser = argparse.ArgumentParser(description='CPU security')
    parser.add_argument('--exp_name', type=str, default='nothing')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--decay', type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--attack_type', type=str, default='FR')
    parser.add_argument('--data_root', type=str, default='/home/jaewon/IITP/datas')
    parser.add_argument('--log_root', type=str, default='./logs')
    parser.add_argument('--split_ratio', type=float, default=0.7, help="ratio of train/validation set from each file")
    parser.add_argument('--max_num', type=int, default=-1, help="maximum num of data in each file (if -1, use all the data)")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--emb_dim', type=int, default=16)
    # parser.add_argument('--load_pickle', action='store_true', help="load pickle file instead of csv file for quick loading")
    args = parser.parse_args()
    arg_dict = vars(args)
    
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        
    save_path = os.path.join(arg_dict['log_root'],arg_dict['exp_name'])
    os.makedirs(save_path, exist_ok=True)
    train_set, val_set = load_dataset(arg_dict, device)

    model = RNN(arg_dict['emb_dim']).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    with tqdm(total=arg_dict['epoch']) as pbar:
        for epoch in range(arg_dict['epoch']):
            total_loss, total_score, count = 0, 0, 0
            for datas, labels in train_set:
                hidden = torch.randn(1,1, arg_dict['emb_dim']).to(device)
                for data, label in zip(datas, labels):
                    count += 1
                    data = data.unsqueeze(1).to(device)
                    label = label.to(device)
                    hidden = hidden.detach()
                    train_score, train_loss, hidden = train(model, optimizer, criterion, data, label, hidden)
                    total_loss += train_loss
                    total_score += train_score

            print("Train || Epoch : {}  Loss : {}   Score : {}".format(epoch+1, total_loss/count, total_score/count))

            right = {c:0 for c in val_set.keys()}
            wrong = {c:0 for c in val_set.keys()}
            metrics = {c:{'tp':0, 'tn':0, 'fp':0, 'fn':0} for c in val_set.keys()}

            for c, c_datas in val_set.items():
                for datas, labels in zip(*c_datas):
                    hidden = torch.randn(1,1,arg_dict['emb_dim']).to(device)
                    for data, label in zip(datas, labels):
                        data = data.unsqueeze(1).to(device)
                        label = label.to(device)
                        hidden = hidden.detach()
                        correct, total, test_loss, hidden = test(model, criterion, data, label, hidden)
                        
                        right[c] += correct
                        wrong[c] += (total - correct)
                        if label[0].item() == 0:
                            metrics[c]['tn'] += correct
                            metrics[c]['fp'] += (total-correct)
                        else:
                            metrics[c]['tp'] += correct
                            metrics[c]['fn'] += (total-correct)
                            
            print("Right : {}, Wrong : {}" .format(sum(right.values()), sum(wrong.values())))
            print("==================== Right ====================")
            print("Low : {}, Mid : {}, High : {}, NoLoad : {}".format(right['low'], right['mid'], right['high'], right['no_load']))
            print("==================== Wrong ====================")
            print("Low : {}, Mid : {}, High : {}, NoLoad : {}".format(wrong['low'], wrong['mid'], wrong['high'], wrong['no_load']))
            print("==================== Score ====================")
            print("Low : {}, Mid : {}, High : {}, NoLoad : {}".format(
                        right['low']/max(1,(right['low']+wrong['low'])), 
                        right['mid']/max(1,(right['mid']+wrong['mid'])), 
                        right['high']/max(1,(right['high']+wrong['high'])), 
                        right['no_load']/max(1,(right['no_load']+wrong['no_load']))))
            print("==================== Precision ==================")
            print("Low : {}, Mid : {}, High : {}".format(
            metrics['low']['tp']/(metrics['low']['tp']+metrics['low']['fp']+1), 
            metrics['mid']['tp']/(metrics['mid']['tp']+metrics['mid']['fp']+1),
            metrics['high']['tp']/(metrics['high']['tp']+metrics['high']['fp']+1)))
            print("==================== Recall ==================")
            print("Low : {}, Mid : {}, High : {}".format(
            metrics['low']['tp']/(metrics['low']['tp']+metrics['low']['fn']+1), 
            metrics['mid']['tp']/(metrics['mid']['tp']+metrics['mid']['fn']+1),
            metrics['high']['tp']/(metrics['high']['tp']+metrics['high']['fn']+1)))
            
            torch.save(model.state_dict(), os.path.join(save_path,"model.pt"))

            pbar.update(1)
        
if __name__ == '__main__':
    main()


