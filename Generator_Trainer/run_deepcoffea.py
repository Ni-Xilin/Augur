import argparse
import os
import torch
from exp.exp_main_deepcoffea import Exp_Main
import random
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deepcoffea')
    # random seed
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')
    
    # basic configbatch_size
    parser.add_argument('--is_training', type=int,  default=0, help='status')
    parser.add_argument('--model_id', type=str,  default='deepcoffea', help='model id')
    parser.add_argument('--model', type=str,  default='PatchTST')
    parser.add_argument('--target_model', type=str,  default='Deepcoffea',
                        help='model name, options: [Deepcorr100,Deepcorr300,Deepcoffea]')
    parser.add_argument('--adv_type', type=str,  default='time_and_size',
                    help='adv type, options: [time,size,time and size]')
    parser.add_argument('--target_model_path', type=str, default='./target_model/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256', help='target model path')
    
    # data loader
    parser.add_argument('--data', type=str,  default='Deepcoffea', help='dataset name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256', help='location of model checkpoints')
    parser.add_argument('--data_path', type=str, default='./target_model/deepcoffea/dataset/CrawlE_Proc', help='data file path')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=150, help='input sequence length') #100->150->200
    # parser.add_argument('--label_len', type=int, default=10, help='start token length')
    parser.add_argument('--pred_len', type=int, default=70, help='prediction sequence length') #50->70->90
    
    # PatchTST
    parser.add_argument('--head_dropout', type=float, default=0.0, help='flatten head dropout')
    parser.add_argument('--patch_len', type=int, default=10, help='patch length')
    parser.add_argument('--stride', type=int, default=70, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=1, help='individual head; True 1 False 0')  
    parser.add_argument('--depth', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor')
    parser.add_argument('--n_layers', type=int, default=2, help='number of layers for each block')
    parser.add_argument('--att_dropout', type=float, default=0.0, help='head dropout')
    
    # Formers 
    # parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=2, help='feature num') 
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    #test和train的batch_size不一样,根据数据集大小和显存大小来设置，本机train batch_size=, test batch_size=32
    parser.add_argument('--batch_size', type=int, default=25, help='batch size of train or test input data')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type4', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=2, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='2', help='device ids of multile gpus')
    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print(args)

Exp = Exp_Main

if args.is_training:
        # setting record of experiments
        setting = '{}_{}_{}_sl{}_pl{}_dm{}_nh{}_pal{}_s{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.patch_len,
            args.stride)

        exp = Exp(args)  # set experiments
        start_time = time.time()
        print('start training : {}>'.format(setting))
        exp.train(setting)
        end_time = time.time()
        print(f"运行时间：{end_time - start_time:.2f} 秒")
else:
    setting = '{}_{}_{}_sl{}_pl{}_dm{}_nh{}_pal{}_s{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.patch_len,
            args.stride
            )

    exp = Exp(args)  # set experiments
    print('testing : {}'.format(setting))
    exp.test(setting)
    torch.cuda.empty_cache()
