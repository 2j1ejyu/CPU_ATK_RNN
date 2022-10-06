import argparse
import torch
import torch.nn as nn 
import os
import pandas as pd
import numpy as np

from data import scaler
from model import RNN

import pdb


input_feats = ['IPC', 'Instructions', 'Cycles', 'Started RTM Exec', 'Aborted RTM Exec', 'TX Write Abort']

def test(model, data, hidden):
    model.eval()
    output, hidden = model(data, hidden)
    output = output.squeeze(1).squeeze(1)
    detection = (torch.sigmoid(output)>=0.5).long()

    return detection, hidden

def main():
    parser = argparse.ArgumentParser(description='CPU security')
    parser.add_argument('--data_path', type=str, default='/home/jaewon/IITP/datas')
    parser.add_argument('--ckpt', type=str, default='./logs')
    parser.add_argument('--max_num', type=int, default=-1, help="maximum num of data in each file (if -1, use all the data)")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--emb_dim', type=int, default=16)
    # parser.add_argument('--load_pickle', action='store_true', help="load pickle file instead of csv file for quick loading")
    args = parser.parse_args()
    arg_dict = vars(args)
    
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")


    model = RNN(arg_dict['emb_dim'])
    model.load_state_dict(torch.load(arg_dict['ckpt']))
    model.to(device)

    assert os.path.isfile(arg_dict['data_path'])
    all_data = pd.read_csv(arg_dict['data_path'], index_col=False)
    columns = [[k.strip() for k in all_data.columns].index(feat) for feat in input_feats]
    
    if arg_dict['max_num'] == -1:
        max_num = len(all_data)
    else:
        max_num = arg_dict['max_num']

    hidden = torch.randn(1,1, arg_dict['emb_dim']).to(device)
    index = 0
    detect = 0

    while index < max_num:
        data = torch.from_numpy(all_data.iloc[index,columns].values.astype(np.float32).reshape(None,...))
        data = scaler(data).unsqueeze(1).to(device)
        hidden = hidden.detach()
        detection, hidden = test(model, data, hidden)
        detect += detection.item()
        index += 1
    
    if detect >= max_num/2:
        print("Attacked!")
    else:
        print("Not Attacked!")

                        
if __name__ == '__main__':
    main()


