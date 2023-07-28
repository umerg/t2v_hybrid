import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import random_neq, truncate_pad

class TrainDataset(Dataset):
    def __init__(self, full_data, data, item_n, max_seq_len):
        self.data = data
        self.full_data = full_data
        self.item_n = item_n
        self.max_seq_len = max_seq_len

    def __len__(self):
        #number of users
        return len(self.data)
    
    def __getitem__(self, idx):

        #for any sequence we fins positive and negative samples and truncate as in original
        user = self.data[idx][0]
        pos_i = self.data[idx][1] #pos index

        in_seq = list(map(lambda x: x[0], self.full_data[user][0])) #item data for input seq
        in_time_seq = list(map(lambda x: x[1], self.full_data[user][0])) #item time data for input seq

        out_seq = list(map(lambda x: x[0], self.full_data[user][1]))

        if pos_i == -1:
            pos_i = np.random.randint(0, len(out_seq))

        pos_ = out_seq[pos_i]

        maxlen = self.max_seq_len

        seq = np.zeros([maxlen], dtype=np.int32)
        time = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = pos_
        idx = maxlen - 1

        ts = set(in_seq) | set(out_seq)
        for i in range(len(in_seq)-1, -1, -1):
            seq[idx] = in_seq[i]
            time[idx] = in_time_seq[i]
            if seq[idx] != 0: 
                pos[idx] = nxt
                neg[idx] = random_neq(1, self.item_n + 1, ts)
            nxt = seq[idx]
            idx -= 1
            if idx == -1: break

        return {"input_seq": torch.tensor(seq), "time_seq": torch.tensor(time), "pos_seq": torch.tensor(pos), "neg_seq": torch.tensor(neg)} 

class ValidDataset(Dataset):
    
    def __init__(self, full_data, data, item_n, max_seq_len):
        
        self.data = data
        self.item_n = item_n
        self.max_seq_len = max_seq_len
        self.full_data = full_data
        #scaling/clipping time diff and relation matrix code borrowed from original for consistency

    def __len__(self):
        
        #number of users
        return len(self.data)
    
    def __getitem__(self, idx):

        #for any sequence we fins positive and negative samples and truncate as in original
        user = self.data[idx][0]
        pos_i = self.data[idx][1] #pos index

        in_seq = list(map(lambda x: x[0], self.full_data[user][0])) #item data for input seq
        in_time_seq = list(map(lambda x: x[1], self.full_data[user][0])) #item time data for input seq

        out_seq = list(map(lambda x: x[0], self.full_data[user][1]))
        out_time_seq = list(map(lambda x: x[1], self.full_data[user][1]))

        pos = out_seq[pos_i]
        pos_time = out_time_seq[pos_i]

        maxlen = self.max_seq_len

        seq = truncate_pad(in_seq, self.max_seq_len) #truncate or pad the seq
        time = truncate_pad(in_time_seq, self.max_seq_len)

        neg_seq = [] #negatives for metrics 

        #avoiding all the original seq and pos items while choosing negs
        ts = set(in_seq) | set(out_seq)

        for i in range(500): #500 negatives per pos
            neg = random_neq(1, self.item_n + 1, ts)
            neg_seq.append(neg)
        
        sample_seq = [pos] + neg_seq

        return {"input_seq": torch.tensor(seq), "time_seq": torch.tensor(time), "sample_seq": torch.tensor(sample_seq)}


class TestDataset(Dataset):
    
    def __init__(self, full_data, data, item_n, max_seq_len):
        
        self.data = data
        self.item_n = item_n
        self.max_seq_len = max_seq_len
        self.full_data = full_data
        #scaling/clipping time diff and relation matrix code borrowed from original for consistency

    def __len__(self):
        
        #number of users
        return len(self.data)
    
    def __getitem__(self, idx):

        #for any sequence we fins positive and negative samples and truncate as in original
        user = self.data[idx][0]
        pos_i = self.data[idx][1] #pos index

        in_seq = list(map(lambda x: x[0], self.full_data[user][0])) #item data for input seq
        in_time_seq = list(map(lambda x: x[1], self.full_data[user][0])) #item time data for input seq

        out_seq = list(map(lambda x: x[0], self.full_data[user][1]))
        out_time_seq = list(map(lambda x: x[1], self.full_data[user][1]))

        pos = out_seq[pos_i]
        pos_time = out_time_seq[pos_i]

        maxlen = self.max_seq_len

        seq = truncate_pad(in_seq, self.max_seq_len) #truncate or pad the seq
        time = truncate_pad(in_time_seq, self.max_seq_len)
        
        neg_seq = [] #negatives for metrics 

        #avoiding all the original seq and pos items while choosing negs
        ts = set(in_seq) | set(out_seq)

        for i in range(500): #500 negatives per pos
            neg = random_neq(1, self.item_n + 1, ts)
            neg_seq.append(neg)
        
        sample_seq = [pos] + neg_seq

        return {"input_seq": torch.tensor(seq), "time_seq": torch.tensor(time), "sample_seq": torch.tensor(sample_seq)}