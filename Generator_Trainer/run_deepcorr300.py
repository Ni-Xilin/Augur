import argparse
import os
import torch
from exp.exp_main_deepcorr300 import Exp_Main
import random
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deepcorr300')
    # random seed
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')
    
    # basic configbatch_size
    parser.add_argument('--is_training', type=int,  default=0, help='status')
    parser.add_argument('--model_id', type=str,  default='deepcorr300', help='model id')
    parser.add_argument('--model', type=str,  default='PatchTST',
                        help='model name, options: [PatchTST]')
    parser.add_argument('--target_model', type=str,  default='Deepcorr300',
                        help='model name, options: [Deepcorr100,Deepcorr300]')
    parser.add_argument('--adv_type', type=str,  default='time_and_size',
                    help='adv type, options: [time,size,time and size]')
    parser.add_argument('--target_model_path', type=str, default='./target_model/deepcorr/deepcorr300/', help='target model path')
    parser.add_argument('--flow_size', type=int, default=300, help='flow size')
    # data loader
    parser.add_argument('--data', type=str,  default='Deepcorr300', help='dataset name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/deepcorr300/', help='location of model checkpoints')
    parser.add_argument('--data_path', type=str, default='./target_model/deepcorr/dataset/', help='data file path')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length') # 64-> 96 -> 128
    # parser.add_argument('--label_len', type=int, default=10, help='start token length')
    parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length') # 32 ->48 -> 64
    
    # PatchTST
    parser.add_argument('--head_dropout', type=float, default=0.0, help='flatten head dropout')
    parser.add_argument('--patch_len', type=int, default=4, help='patch length')
    parser.add_argument('--stride', type=int, default=48, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=1, help='individual head; True 1 False 0')
    parser.add_argument('--Dtarget', type=int, default=1, help='adv type(define loss)')
    parser.add_argument('--depth', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor')
    parser.add_argument('--n_layers', type=int, default=2, help='number of layers for each block')
    parser.add_argument('--att_dropout', type=float, default=0.0, help='head dropout')
    # RevIN
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    
    # Formers 
    # parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=4, help='feature num') 
    # parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    # parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.1, help='pct_start')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='main gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='1', help='device ids of multile gpus')
    #用于计算和打印给定模型的参数数量和计算复杂度（FLOPs），用于评估模型的复杂性和性能
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fix_seed)


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
    setting = '{}_{}_{}_sl{}_pl{}_dm{}_nh{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.n_heads,)

    exp = Exp(args)  # set experiments
    exp.train(setting)
        
else:
    exp = Exp(args)  # set experiments
    exp.test()
    # setting = '{}_{}_{}_sl{}_pl{}_dm{}_nh{}_{}'.format(
    #         args.model_id,
    #         args.model,
    #         args.data,
    #         args.seq_len,
    #         args.pred_len,
    #         args.d_model,
    #         args.n_heads,
    #         ii)

    # exp = Exp(args)  # set experiments
    # print('testing : {}'.format(setting))
    # exp.test(setting, test=1)
    # torch.cuda.empty_cache()
