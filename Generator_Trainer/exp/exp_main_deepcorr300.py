from data_provider.data_factory import data_provider
from exp.exp_basic_deepcorr import Exp_Basic
from target_model import Deepcorr300  
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from torch.utils.data import DataLoader, TensorDataset
from generator.generator import TSTGenerator
from torch.utils.data import Dataset, DataLoader
from utils.metrics import metric
import numpy as np
import torch.nn.functional as F
import torch
import pathlib
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def L2_norm_distance(x, y):
    distance = torch.linalg.norm(x - y, ord=2,dim=(-1))
    return distance

class DeepCorrTSTDataset(Dataset):
    def __init__(self, data_path, flag, args):
        assert flag in ['train', 'test', 'val']
        self.args = args
        self.flow_size = args.flow_size
        
        self.dataset = []
        all_runs = {'8872':'192.168.122.117', '8802':'192.168.122.117', '8873':'192.168.122.67', '8803':'192.168.122.67',
                    '8874':'192.168.122.113', '8804':'192.168.122.113', '8875':'192.168.122.120',
                    '8876':'192.168.122.30', '8877':'192.168.122.208', '8878':'192.168.122.58'}
        for name in all_runs:
            self.dataset += pickle.load(open(f'{data_path}{name}_tordata300.pickle', 'rb'))
        val_index_path = f'{data_path}val_index300.pickle'
        test_index_path = f'{data_path}test_index300.pickle'
        
        if not (os.path.exists(val_index_path) and os.path.exists(test_index_path)):
            print("Index files not found. Creating and saving new splits...")
            len_tr = len(self.dataset)
            train_ratio = float(len_tr - 3000) / float(len_tr)
            rr = list(range(len(self.dataset)))
            np.random.shuffle(rr)
            train_index = rr[:int(len_tr * train_ratio)]
            val_index = rr[int(len_tr * train_ratio):-2000]
            test_index = rr[-2000:-1000]
            pickle.dump(train_index, open(f'{data_path}train_index300.pickle', 'wb'))
            pickle.dump(val_index, open(val_index_path, 'wb'))
            pickle.dump(test_index, open(test_index_path, 'wb'))
        
        if flag == 'train':
            self.indices = pickle.load(open(f'{data_path}train_index300.pickle', 'rb'))
        elif flag == 'val':
            self.indices = pickle.load(open(val_index_path, 'rb'))[:1000]
        else: # test
            self.indices = pickle.load(open(test_index_path, 'rb'))[:1000]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        sample_dict = self.dataset[data_idx]
        
        def pad_or_truncate(data, length):
            return data[:length] if len(data) >= length else data + [0.0] * (length - len(data))

        single_sample = torch.zeros(8, self.flow_size)
        single_sample[0, :] = torch.tensor(pad_or_truncate(sample_dict['here'][0]['<-'], self.flow_size)) * 1000.0
        single_sample[1, :] = torch.tensor(pad_or_truncate(sample_dict['there'][0]['->'], self.flow_size)) * 1000.0
        single_sample[2, :] = torch.tensor(pad_or_truncate(sample_dict['there'][0]['<-'], self.flow_size)) * 1000.0
        single_sample[3, :] = torch.tensor(pad_or_truncate(sample_dict['here'][0]['->'], self.flow_size)) * 1000.0
        single_sample[4, :] = torch.tensor(pad_or_truncate(sample_dict['here'][1]['<-'], self.flow_size)) / 1000.0
        single_sample[5, :] = torch.tensor(pad_or_truncate(sample_dict['there'][1]['->'], self.flow_size)) / 1000.0
        single_sample[6, :] = torch.tensor(pad_or_truncate(sample_dict['there'][1]['<-'], self.flow_size)) / 1000.0
        single_sample[7, :] = torch.tensor(pad_or_truncate(sample_dict['here'][1]['->'], self.flow_size)) / 1000.0
        
        row = [0, 3, 4, 7]
        batch_num = int((self.flow_size - self.args.seq_len - self.args.pred_len) / self.args.stride + 1)
        
        seq = torch.zeros(4, batch_num, self.args.seq_len)
        pred = torch.zeros(4, batch_num, self.args.pred_len)
        
        for j in range(4):
            for i in range(batch_num):
                seq[j, i, :] = single_sample[row[j], i * self.args.stride : i * self.args.stride + self.args.seq_len]
                pred[j, i, :] = single_sample[row[j], i * self.args.stride + self.args.seq_len : i * self.args.stride + self.args.seq_len + self.args.pred_len]
        
        return seq, pred, single_sample


class ConvertedDeepCorrDataset(Dataset):
    """
    用于加载和预处理转换后的DeepCorr pickle文件的数据集类。
    """
    def __init__(self, seq_len, pred_len, stride):
        pickle_path = "datasets_convert/deepcorr_test_from_deepcoffea.pickle" 
        print(f"正在从 {pickle_path} 加载数据...")
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)
        print(f"成功加载 {len(self.dataset)} 条样本。")

        self.flow_size = 5000
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample_dict = self.dataset[index]
        # 将DeepCorr格式的数据转换为 (8, flow_size) 的张量
        single_sample = torch.zeros(8, self.flow_size)
        def pad_or_truncate(data, length):
            return data[:length] if len(data) >= length else data + [0.0] * (length - len(data))
        # here (tor)
        single_sample[0, :] = torch.tensor(pad_or_truncate(sample_dict['here'][0]['<-'], self.flow_size)) * 1000.0
        single_sample[3, :] = torch.tensor(pad_or_truncate(sample_dict['here'][0]['->'], self.flow_size)) * 1000.0
        single_sample[4, :] = torch.tensor(pad_or_truncate(sample_dict['here'][1]['<-'], self.flow_size)) / 1000.0
        single_sample[7, :] = torch.tensor(pad_or_truncate(sample_dict['here'][1]['->'], self.flow_size)) / 1000.0
        
        # there (exit)
        single_sample[1, :] = torch.tensor(pad_or_truncate(sample_dict['there'][0]['->'], self.flow_size)) * 1000.0
        single_sample[2, :] = torch.tensor(pad_or_truncate(sample_dict['there'][0]['<-'], self.flow_size)) * 1000.0
        single_sample[5, :] = torch.tensor(pad_or_truncate(sample_dict['there'][1]['->'], self.flow_size)) / 1000.0
        single_sample[6, :] = torch.tensor(pad_or_truncate(sample_dict['there'][1]['<-'], self.flow_size)) / 1000.0

        # 将流量分割为 seq 和 pred 批次，仅针对 'here' 部分的通道
        row = [0, 3, 4, 7]
        num_sub_batches = (self.flow_size - self.seq_len - self.pred_len) // self.stride + 1
        
        seq = torch.zeros(4, num_sub_batches, self.seq_len)
        pred = torch.zeros(4, num_sub_batches, self.pred_len)
        
        for j, channel_idx in enumerate(row):
            for i in range(num_sub_batches):
                start_idx = i * self.stride
                seq_end_idx = start_idx + self.seq_len
                pred_end_idx = seq_end_idx + self.pred_len
                
                seq[j, i, :] = single_sample[channel_idx, start_idx:seq_end_idx]
                pred[j, i, :] = single_sample[channel_idx, seq_end_idx:pred_end_idx]
        
        return seq, pred, single_sample
    
    
    
class Loss_targeted(nn.Module):
    def __init__(self,beta=1, alpha =3, gamma=0.9): 
        super(Loss_targeted, self).__init__()
        self.beta = beta 
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, target_outputs,target_labels,batch_x,adv_batch_flow):

        criterion = nn.BCEWithLogitsLoss()
        row_time =[0,3]
        row_size =[4,7]
        time_L2_distance = L2_norm_distance(batch_x[:,:,row_time,:],adv_batch_flow[:,:,row_time,:])
        size_L2_distance = L2_norm_distance(batch_x[:,:,row_size,:],adv_batch_flow[:,:,row_size,:])
        time_rate = (time_L2_distance/(torch.linalg.norm(batch_x[:,:,row_time,:], ord=2,dim=(-1)))).mean()
        size_rate = (size_L2_distance/(torch.linalg.norm(batch_x[:,:,row_size,:], ord=2,dim=(-1)))).mean()

        # loss = criterion(target_outputs,target_labels)
        loss = self.beta*criterion(target_outputs,target_labels)+self.alpha*time_rate + self.gamma*size_rate
        # print("loss:{:4f},label loss:{:4f}, time : {:4f},size : {:4f}".format(loss,criterion(target_outputs,target_labels), time_rate.item(),size_rate.item()))
        # loss = criterion(target_outputs,target_labels)
        # loss = self.beta*self.size_L2_distance
        return loss,criterion(target_outputs,target_labels).item(),time_rate.item(),size_rate.item(),self.alpha,self.beta,self.gamma

        
class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args = args
    def _build_model(self):
        target_model_dict = {
            'Deepcorr300': Deepcorr300,
        }
        Target_model = target_model_dict[self.args.target_model].Model().float()
        Generator = TSTGenerator(seq_len=self.args.seq_len, 
                                patch_len=self.args.patch_len,
                                pred_len=self.args.pred_len,
                                feat_dim=self.args.enc_in, 
                                depth=self.args.depth, 
                                scale_factor=self.args.scale_factor, 
                                n_layers=self.args.n_layers, 
                                d_model=self.args.d_model, 
                                n_heads=self.args.n_heads,
                                individual=self.args.individual, 
                                d_k=None, d_v=None, 
                                d_ff=self.args.d_ff, 
                                norm='BatchNorm', 
                                attn_dropout=self.args.att_dropout, 
                                head_dropout=self.args.head_dropout, 
                                act=self.args.activation,pe='zeros', 
                                learn_pe=True,pre_norm=False, 
                                res_attention=False, 
                                store_attn=False)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            Target_model = nn.DataParallel(Target_model, device_ids=self.args.device_ids)
            Generator = nn.DataParallel(Generator, device_ids=self.args.device_ids)
            
        return Target_model,Generator

    # def _get_data(self, flag):
        
    #     data_set, data_loader = data_provider(self.args, flag)
    #     return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.generator.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):

        criterion = Loss_targeted()

        return criterion


    def train(self, setting):
        print("Creating DataLoaders with multi-processing enabled...")
        train_dataset = DeepCorrTSTDataset(data_path=self.args.data_path, flag='train', args=self.args)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                  num_workers=self.args.num_workers, drop_last=True)
        
        vali_dataset = DeepCorrTSTDataset(data_path=self.args.data_path, flag='val', args=self.args)
        vali_loader = DataLoader(vali_dataset, batch_size=self.args.batch_size, shuffle=False,
                                 num_workers=self.args.num_workers, drop_last=True)
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)

        model_optim = optim.Adam(self.generator.parameters(), lr=self.args.learning_rate)
        criterion = Loss_targeted()
        train_steps = len(train_loader)
        scheduler_oclr = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate,
                                            )
        state_dict = torch.load(f"{self.args.target_model_path}tor_199_epoch23_acc0.82dict.pth", map_location=self.device)
        self.target_model.load_state_dict(state_dict)
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False
        
        print(f"Starting training for {self.args.train_epochs} epochs...")
        for epoch in range(self.args.train_epochs):
            self.generator.train()
            epoch_time = time.time()
            epoch_losses, epoch_label_losses, epoch_time_ratios, epoch_size_ratios = [], [], [], []
            
            for i, batch in enumerate(train_loader):
                model_optim.zero_grad()
                batch_seq, batch_pred, batch_original = batch
                
                batch_size, num_channels, num_sub_batches, _ = batch_seq.shape
                z = batch_seq.permute(0, 2, 3, 1).reshape(batch_size * num_sub_batches, self.args.seq_len, num_channels).permute(0, 2, 1).to(self.device)
                batch_x_pred = batch_pred.permute(0, 2, 3, 1).reshape(batch_size * num_sub_batches, self.args.pred_len, num_channels).to(self.device)
                batch_x_original_gpu = batch_original.clone().unsqueeze(1).to(self.device)

                delta = self.generator(z).permute(0, 2, 1)
                abs_x = torch.abs(batch_x_pred)
                delta = torch.abs(delta)
                constraint_delta = delta.clone()
                constraint_time = torch.clamp(delta[:, :, [0, 1]], min=0 * abs_x[:, :, [0, 1]])
                constraint_size = torch.clamp(delta[:, :, [2, 3]], min=0 * abs_x[:, :, [2, 3]])
                delta[:, :, [0, 1]], delta[:, :, [2, 3]] = constraint_time, constraint_size
                constraint_delta[:,:,[0,1]] = constraint_time
                constraint_delta[:,:,[2,3]] = constraint_size
                adv_batch_x_pred = batch_x_pred + constraint_delta
                
                adv_batch_x_pred_reshaped = adv_batch_x_pred.reshape(batch_size, num_sub_batches, self.args.pred_len, num_channels).permute(0, 3, 1, 2)
                adv_batch_flow = batch_x_original_gpu.clone()
                row = [0, 3, 4, 7]
                for j in range(num_channels):
                    for k in range(num_sub_batches):
                        start = k * self.args.stride + self.args.seq_len
                        end = start + self.args.pred_len
                        adv_batch_flow[:, 0, row[j], start:end] = adv_batch_x_pred_reshaped[:, j, k, :]
                
                target_outputs = self.target_model(adv_batch_flow, dropout=0.0)
                target_labels = torch.zeros_like(target_outputs).float().to(self.device)
                
                loss, label_loss, time_ratio, size_ratio,alpha,beta,gamma = criterion(target_outputs, target_labels, batch_x_original_gpu, adv_batch_flow)
                print(f"Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Label Loss: {label_loss:.4f}, Time Ratio: {time_ratio:.4f}, Size Ratio: {size_ratio:.4f}")
                if(i+1) % 40 == 0 and time_ratio<0.16 and size_ratio<0.16:
                    acc_original,acc_adv,avg_time_ratio,avg_size_ratio = self.vali(vali_loader, criterion)           
                    if(avg_time_ratio < 0.15 and avg_size_ratio < 0.15 and acc_adv < 0.1):
                        torch.save({
                                    'epoch': epoch + 1,
                                    'Time L2 rate':avg_time_ratio,
                                    'Size L2 rate':avg_size_ratio,
                                    'vali_adv_correct_rate':acc_adv,
                                    'vali_correct_rate': acc_original,
                                    'generator_state_dict': self.generator.state_dict(),
                                    'optimizer_state_dict': model_optim.state_dict(),
                                    'alpha': alpha,
                                    'beta': beta,
                                    'gamma': gamma,
                                }, os.path.join(path, f'generator_checkpoint_{epoch + 1}_acc_{acc_original:.3f}_Gacc{acc_adv:.4f}_advipd{avg_time_ratio:.3f}_advsize{avg_size_ratio:.3f}.pth'))
                loss.backward()
                model_optim.step()

                epoch_losses.append(loss.item())
                epoch_label_losses.append(label_loss)
                epoch_time_ratios.append(time_ratio)
                epoch_size_ratios.append(size_ratio)
            adjust_learning_rate(model_optim, scheduler_oclr, epoch + 1, self.args, printout=True)
            avg_loss = np.mean(epoch_losses)
            avg_label_loss = np.mean(epoch_label_losses)
            avg_time_ratio = np.mean(epoch_time_ratios)
            avg_size_ratio = np.mean(epoch_size_ratios)

            print(f"\nEpoch: {epoch + 1} | Cost Time: {time.time() - epoch_time:.2f}s")
            print(f"Train Loss: {avg_loss:.4f}, Label Loss: {avg_label_loss:.4f}, Time Ratio: {avg_time_ratio:.4f}, Size Ratio: {avg_size_ratio:.4f}")
            
            acc_original,acc_adv,avg_time_ratio,avg_size_ratio = self.vali(vali_loader, criterion)
            
            if(epoch == 4):
                torch.save({
                            'epoch': epoch + 1,
                            'Time L2 rate':avg_time_ratio,
                            'Size L2 rate':avg_size_ratio,
                            'vali_adv_correct_rate':acc_adv,
                            'vali_correct_rate': acc_original,
                            'generator_state_dict': self.generator.state_dict(),
                            'optimizer_state_dict': model_optim.state_dict(),
                            'loss': avg_loss,
                            'alpha': alpha,
                            'beta': beta,
                            'gamma': gamma,
                        }, os.path.join(path, f'generator_checkpoint_{epoch + 1}_acc_{acc_original:.3f}_Gacc{acc_adv:.4f}_advipd{avg_time_ratio:.3f}_advsize{avg_size_ratio:.3f}.pth'))
            # self.generator.load_state_dict(torch.load(best_model_path))
        return self.generator
    
    def vali(self, vali_loader, criterion):
        self.generator.eval()
        self.target_model.eval()
        
        total, correct, adv_correct = 0, 0, 0
        total_time_ratios, total_size_ratios = [], []
        
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_seq, batch_pred, batch_original = batch
                
                batch_size, num_channels, num_sub_batches, _ = batch_seq.shape
                z = batch_seq.permute(0, 2, 3, 1).reshape(batch_size * num_sub_batches, self.args.seq_len, num_channels).permute(0, 2, 1).to(self.device)
                batch_x_pred = batch_pred.permute(0, 2, 3, 1).reshape(batch_size * num_sub_batches, self.args.pred_len, num_channels).to(self.device)
                batch_x_original_gpu = batch_original.clone().unsqueeze(1).to(self.device)

                delta = self.generator(z).permute(0, 2, 1)
                abs_x = torch.abs(batch_x_pred)
                delta = torch.abs(delta)
                #避免扰动padding的0
                constraint_time = torch.clamp(delta[:, :, [0, 1]], min=0 * abs_x[:, :, [0, 1]], max=100*abs_x[:, :, [0, 1]])
                constraint_size = torch.clamp(delta[:, :, [2, 3]], min=0 * abs_x[:, :, [2, 3]],max=100*abs_x[:, :, [2, 3]])
                delta[:, :, [0, 1]], delta[:, :, [2, 3]] = constraint_time, constraint_size
                adv_batch_x_pred = torch.sign(batch_x_pred) * (abs_x + delta)

                adv_batch_x_pred_reshaped = adv_batch_x_pred.reshape(batch_size, num_sub_batches, self.args.pred_len, num_channels).permute(0, 3, 1, 2)
                adv_batch_flow = batch_x_original_gpu.clone()
                row = [0, 3, 4, 7]
                for j in range(num_channels):
                    for k in range(num_sub_batches):
                        start = k * self.args.stride + self.args.seq_len
                        end = start + self.args.pred_len
                        adv_batch_flow[:, 0, row[j], start:end] = adv_batch_x_pred_reshaped[:, j, k, :]
                
                true_labels = torch.ones(batch_size, 1).to(self.device)
                outputs_original = self.target_model(batch_x_original_gpu, dropout=0.0)
                outputs_adv = self.target_model(adv_batch_flow, dropout=0.0)
                
                _, _, time_ratio, size_ratio,_,_,_ = criterion(outputs_adv, true_labels, batch_x_original_gpu, adv_batch_flow)
                total_time_ratios.append(time_ratio)
                total_size_ratios.append(size_ratio)
                
                total += batch_size
                correct += ((torch.sigmoid(outputs_original) > 0.1) == (true_labels ==1)).sum().item()
                adv_correct += ((torch.sigmoid(outputs_adv) > 0.1) == (true_labels  ==1)).sum().item()

        acc_original = correct / total if total > 0 else 0
        acc_adv = adv_correct / total if total > 0 else 0
        avg_time_ratio = np.mean(total_time_ratios)
        avg_size_ratio = np.mean(total_size_ratios)

        print("\n--- Validation Results ---")
        print(f"Accuracy on Original Data: {acc_original:.4f}")
        print(f"Accuracy on Adversarial Data: {acc_adv:.4f}")
        print(f"Average Time L2 Ratio: {avg_time_ratio:.4f}")
        print(f"Average Size L2 Ratio: {avg_size_ratio:.4f}")
               
        self.generator.train()

        return acc_original,acc_adv,avg_time_ratio,avg_size_ratio
        
    
    
    def test(self):
        self.generator.eval()
        self.target_model.eval()
        
        total_time_ratios, total_size_ratios = [], []
        
        test_dataset = DeepCorrTSTDataset(data_path=self.args.data_path, flag='test', args=self.args)
        # test_dataset = ConvertedDeepCorrDataset(seq_len=self.args.seq_len, pred_len=self.args.pred_len, stride=self.args.stride)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False,
                                 num_workers=self.args.num_workers, drop_last=True)

        checkpoint = torch.load('checkpoints/deepcorr300/deepcorr300_PatchTST_Deepcorr300_sl96_pl48_dm512_nh8/generator_checkpoint_9_acc_0.878_Gacc0.0423_advipd0.137_advsize0.009.pth', map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        adv_samples = []
        original_samples = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_seq, batch_pred, batch_original = batch
                
                batch_size, num_channels, num_sub_batches, _ = batch_seq.shape
                z = batch_seq.permute(0, 2, 3, 1).reshape(batch_size * num_sub_batches, self.args.seq_len, num_channels).permute(0, 2, 1).to(self.device)
                batch_x_pred = batch_pred.permute(0, 2, 3, 1).reshape(batch_size * num_sub_batches, self.args.pred_len, num_channels).to(self.device)
                batch_x_original_gpu = batch_original.clone().unsqueeze(1).to(self.device)

                delta = self.generator(z).permute(0, 2, 1)
                abs_x = torch.abs(batch_x_pred)
                delta = torch.abs(delta)
                #避免扰动padding的0
                constraint_time = torch.clamp(delta[:, :, [0, 1]], min=0 * abs_x[:, :, [0, 1]], max=100*abs_x[:, :, [0, 1]])
                constraint_size = torch.clamp(delta[:, :, [2, 3]], min=0 * abs_x[:, :, [2, 3]],max=100*abs_x[:, :, [2, 3]])
                delta[:, :, [0, 1]], delta[:, :, [2, 3]] = constraint_time, constraint_size
                adv_batch_x_pred = torch.sign(batch_x_pred) * (abs_x + delta)

                adv_batch_x_pred_reshaped = adv_batch_x_pred.reshape(batch_size, num_sub_batches, self.args.pred_len, num_channels).permute(0, 3, 1, 2)
                adv_batch_flow = batch_x_original_gpu.clone()
                row = [0, 3, 4, 7]
                for j in range(num_channels):
                    for k in range(num_sub_batches):
                        start = k * self.args.stride + self.args.seq_len
                        end = start + self.args.pred_len
                        adv_batch_flow[:, 0, row[j], start:end] = adv_batch_x_pred_reshaped[:, j, k, :]
                
                adv_samples.append(adv_batch_flow)
                original_samples.append(batch_x_original_gpu)
                
                time_L2_distance = L2_norm_distance(batch_x_original_gpu[:,:,[0, 3],:],adv_batch_flow[:,:,[0, 3],:])
                size_L2_distance = L2_norm_distance(batch_x_original_gpu[:,:,[4, 7],:],adv_batch_flow[:,:,[4, 7],:])
                time_ratio = (time_L2_distance/(torch.linalg.norm(batch_x_original_gpu[:,:,[0, 3],:], ord=2,dim=(-1)))).mean()
                size_ratio = (size_L2_distance/(torch.linalg.norm(batch_x_original_gpu[:,:,[4, 7],:], ord=2,dim=(-1)))).mean()
                
                total_time_ratios.append(time_ratio.cpu())
                total_size_ratios.append(size_ratio.cpu())
        avg_time_ratio = np.mean(total_time_ratios)
        avg_size_ratio = np.mean(total_size_ratios)
        print("\n--- Test Results ---")
        print(f"Average Time L2 Ratio: {avg_time_ratio:.4f}")
        print(f"Average Size L2 Ratio: {avg_size_ratio:.4f}")
    
        adv_samples = torch.cat(adv_samples, dim=0)
        original_samples = torch.cat(original_samples, dim=0)

        # 转换为numpy数组
        result_fpath = pathlib.Path(f'target_model/deepcorr/deepcorr300/Gbase/Gdeepcorr_advsamples_time{avg_time_ratio:.4f}_size{avg_size_ratio:.4f}.p')
        with open(result_fpath, "wb") as fp:
            results = {
                "adv_samples": adv_samples,
                "original_samples": original_samples,
                "time_ratios": avg_time_ratio,
                "size_ratios": avg_size_ratio,
            }
            pickle.dump(results, fp)
        # result_fpath = pathlib.Path(f'datasets_convert/Gconverted_advsamples_time{avg_time_ratio:.4f}_size{avg_size_ratio:.4f}.p')
        # with open(result_fpath, "wb") as fp:
        #     results = {
        #         "adv_samples": adv_samples,
        #         "original_samples": original_samples,
        #         "time_ratios": avg_time_ratio,
        #         "size_ratios": avg_size_ratio,
        #     }
        #     pickle.dump(results, fp)