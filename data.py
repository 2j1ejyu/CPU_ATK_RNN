import os
import torch
import pickle
import random
import pdb
from glob import glob
import pandas as pd
import numpy as np
import itertools

input_feats = ['IPC', 'Instructions', 'Cycles', 'Started RTM Exec', 'Aborted RTM Exec', 'TX Write Abort']
cpu_usages = ['no_load','low','mid','high']

def load_data(arg_dict):
    atk_type = arg_dict['attack_type']
    data_root = arg_dict['data_root']
    # ext = 'pkl' if arg_dict['load_pickle'] else 'csv'
    B = arg_dict['batch_size']
    ratio = arg_dict['split_ratio']

    train_datas = {c:[] for c in cpu_usages}
    train_labels = {c:[] for c in cpu_usages}
    val_datas = {c:[] for c in cpu_usages}
    val_labels = {c:[] for c in cpu_usages}

    for c in cpu_usages:
        for l in ['0','1']:
            files_path = glob(os.path.join(data_root,atk_type,c,l,"*.csv"))
            for path in files_path:
                name = os.path.basename(path)
                print("loading "+ name)
                data = pd.read_csv(path, index_col=False)
                indices = [[k.strip() for k in data.columns].index(feat) for feat in input_feats]

                if arg_dict['max_num'] == -1:
                    max_num = len(data)
                else:
                    max_num = min(arg_dict['max_num'],len(data))

                train_num = int(max_num*ratio)
                val_num = max_num - train_num

                temp_datas = []
                temp_labels = []
                ### train ###
                for Iter in range(train_num//B + 1):
                    start = Iter * B
                    end = min(train_num, (Iter+1)*B)
                    if l=='0':
                        label = torch.zeros(end-start)
                    else:
                        label = torch.ones(end-start)
                    temp_datas.append(scaler(torch.from_numpy(data.iloc[start:end,indices].values.astype(np.float32))))
                    temp_labels.append(label)
                train_datas[c].append(temp_datas)
                train_labels[c].append(temp_labels)

                temp_datas = []
                temp_labels = []
                ### val ###
                for Iter in range(val_num//B + 1):
                    start = Iter * B + train_num
                    end = min(val_num, (Iter+1)*B) + train_num
                    if l=='0':
                        label = torch.zeros(end-start)
                    else:
                        label = torch.ones(end-start)
                    temp_datas.append(scaler(torch.from_numpy(data.iloc[start:end,indices].values.astype(np.float32))))
                    temp_labels.append(label)
                val_datas[c].append(temp_datas)
                val_labels[c].append(temp_labels)

    return train_datas, train_labels, val_datas, val_labels

def scaler(data, atk_type='PA'):
    new_data = data.clone()
    if atk_type=="PA":
        new_data[:, 0] = (new_data[:, 0]-(1.11))/(1.35-0.96)
        new_data[:, 1] = (data[:, 1]-(1.4264e+7))/(1.642038e+7 - 1.172859e+7)
        new_data[:, 2] = (data[:, 2]-(1.191183e+7))/(1.468553e+7 - 9.989132e+6)
        new_data[:, 5] = (data[:, 5]-(1.191153e+7))/(1.468578e+7 - 1.470713e+6)
    elif atk_type=="FR":
        new_data[:, 0] = (new_data[:, 0]-(6.6e-1))/(4.1e-01)
        new_data[:, 1] = (data[:, 1]-(1.259074e+7))/5021368
        new_data[:, 2] = (data[:, 2]-(1.757167e+7))/8262030
        new_data[:, 5] = (data[:, 5]-(1.757163e+7))/8262000

    return new_data

def load_dataset(arg_dict, train=True):
    train_datas, train_labels, val_datas, val_labels = load_data(arg_dict)
    train_set = []
    for c in train_datas.keys():
        for i in range(len(train_datas[c])):
            train_set.append((train_datas[c][i], train_labels[c][i]))
    assert len(train_set) != 0
    random.shuffle(train_set)

    val_set = {}
    for c in val_datas.keys():
        val_set[c] = (val_datas[c],val_labels[c])
    
    return train_set, val_set

