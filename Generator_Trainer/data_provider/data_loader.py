import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader,Sampler
from tqdm import tqdm
import pickle
import pathlib
import warnings

warnings.filterwarnings('ignore')


class Deepcorr300(Dataset):
    def __init__(self,data_path, flag='train'):
        # size [seq_len, label_len, pred_len]    
        # init
        assert flag in ['train', 'test', 'val']
        
        self.set_type = flag
        self.data_path = data_path
        self.flow_size = 300

              
        self.__read_data__()

    def __read_data__(self):
        
        flow_size = self.flow_size
        self.dataset = []
        all_runs={
            '8872':'192.168.122.117',
            '8802':'192.168.122.117',
            '8873':'192.168.122.67',
            '8803':'192.168.122.67',
            '8874':'192.168.122.113',
            '8804':'192.168.122.113',
            '8875':'192.168.122.120',
            '8876':'192.168.122.30',
            '8877':'192.168.122.208',
            '8878':'192.168.122.58'}
        
        #这里flow_size=100与flow_size=300数据集无区别，区别在于取特征特征维度为100还是300
        for name in all_runs:
            self.dataset += pickle.load(open('%s%s_tordata300.pickle' % (self.data_path,name), 'rb'))
        
        if self.set_type == 'train':
            
            len_tr = len(self.dataset)
            train_ratio = float(len_tr - 3000) / float(len_tr)
            rr = list(range(len(self.dataset)))
            np.random.shuffle(rr)

            train_index = rr[:int(len_tr * train_ratio)]
            val_index = rr[int(len_tr * train_ratio):-2000]
            test_index= rr[-2000:-1000]
            
            pickle.dump(val_index, open('%sval_index300.pickle'% (self.data_path), 'wb'))
            pickle.dump(test_index, open('%stest_index300.pickle'% (self.data_path), 'wb'))
  
        elif self.set_type == 'val':
            val_index = pickle.load(open('%sval_index300.pickle'% (self.data_path), 'rb'))[:1000]
            
        else:
            test_index = pickle.load(open('%stest_index300.pickle'% (self.data_path), 'rb'))[:1000]
        
        if self.set_type == 'train':
            self.train_data = torch.zeros((len(train_index),1, 8, flow_size))
            
            j=0
            for i in train_index:
                self.train_data[j, 0, 0,:] = torch.tensor(self.dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0  
                self.train_data[j, 0, 1,:] = torch.tensor(self.dataset[i]['there'][0]['->'][:flow_size]) * 1000.0  
                self.train_data[j, 0, 2,:] = torch.tensor(self.dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0  
                self.train_data[j, 0, 3,:] = torch.tensor(self.dataset[i]['here'][0]['->'][:flow_size]) * 1000.0  
                self.train_data[j, 0, 4,:] = torch.tensor(self.dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0  
                self.train_data[j, 0, 5,:] = torch.tensor(self.dataset[i]['there'][1]['->'][:flow_size]) / 1000.0  
                self.train_data[j, 0, 6,:] = torch.tensor(self.dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0  
                self.train_data[j, 0, 7,:] = torch.tensor(self.dataset[i]['here'][1]['->'][:flow_size]) / 1000.0  
                j = j + 1
                
            self.data_x = self.train_data

        elif self.set_type == 'test':
            self.test_data = torch.zeros((len(test_index),1, 8, flow_size))  
            j=0
            for i in test_index:
                self.test_data[j, 0, 0,:] = torch.tensor(self.dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0  
                self.test_data[j, 0, 1,:] = torch.tensor(self.dataset[i]['there'][0]['->'][:flow_size]) * 1000.0  
                self.test_data[j, 0, 2,:] = torch.tensor(self.dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0  
                self.test_data[j, 0, 3,:] = torch.tensor(self.dataset[i]['here'][0]['->'][:flow_size]) * 1000.0  
                self.test_data[j, 0, 4,:] = torch.tensor(self.dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0  
                self.test_data[j, 0, 5,:] = torch.tensor(self.dataset[i]['there'][1]['->'][:flow_size]) / 1000.0  
                self.test_data[j, 0, 6,:] = torch.tensor(self.dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0  
                self.test_data[j, 0, 7,:] = torch.tensor(self.dataset[i]['here'][1]['->'][:flow_size]) / 1000.0  
                j = j + 1
                
            self.data_x = self.test_data
        else:
            self.val_data = torch.zeros((len(val_index),1, 8, flow_size))  
            j=0
            for i in val_index:
                self.val_data[j, 0, 0,:] = torch.tensor(self.dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0  
                self.val_data[j, 0, 1,:] = torch.tensor(self.dataset[i]['there'][0]['->'][:flow_size]) * 1000.0  
                self.val_data[j, 0, 2,:] = torch.tensor(self.dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0  
                self.val_data[j, 0, 3,:] = torch.tensor(self.dataset[i]['here'][0]['->'][:flow_size]) * 1000.0  
                self.val_data[j, 0, 4,:] = torch.tensor(self.dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0  
                self.val_data[j, 0, 5,:] = torch.tensor(self.dataset[i]['there'][1]['->'][:flow_size]) / 1000.0  
                self.val_data[j, 0, 6,:] = torch.tensor(self.dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0  
                self.val_data[j, 0, 7,:] = torch.tensor(self.dataset[i]['here'][1]['->'][:flow_size]) / 1000.0  
                j = j + 1
                
            self.data_x = self.val_data
                    
            
    def __getitem__(self, index):
        seq_x = self.data_x[index,0,:,:]
        return seq_x

    def __len__(self):
        return len(self.data_x)

    

class DeepCoffeaDataset(Dataset):
    '''
        返回batch_size个会话tor端的正常（待扰动）流量,exit端窗口流量：self.win_tor[:,idx,:],self.win_exit[:,idx,:],包括ipd和size
    '''

    def __init__(self, win_data,train=True):
        
        
        if train:
            mode = "train"
        else:
            mode = "test"
        '''
            v_tor:[n_wins, n_train_samples, tor_len * 2]
            v_exit:[n_wins, n_train_samples, exit_len * 2]  
            v_label:[n_wins, n_train_samples, 1]
            
        '''
        self.win_tor =  win_data[f"{mode}_tor"]
        self.win_exit=  win_data[f"{mode}_exit"]
        
        # session_tor_ipd = session_data['tor_ipd']
        # session_tor_size = session_data['tor_size']
        # session_tor_len = session_data['tor_lengths']#[n_train_samples,n_wins]
        # session_label = session_data['labels']
        
        print(f"Number of (session)examples to {mode}: {self.win_exit.shape[1]}, number of windows: {self.win_exit.shape[0]}")

    def __len__(self):
        return self.win_exit.shape[1]

    def __getitem__(self, idx):
        #返回正常流量
        return idx,self.win_tor[:,idx,:],self.win_exit[:,idx,:]









