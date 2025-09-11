import layers
from layers.PatchTST_backbone import *
from layers.PatchTST_layers import *
import torch 
import torch.nn as nn

class TSTGenerator (nn.Module):
    def __init__(self, 
                 seq_len, 
                 patch_len,
                 pred_len, 
                 feat_dim, 
                 depth=3, 
                 scale_factor=2,
                 n_layers=1, 
                 d_model=128, 
                 n_heads=16, 
                 individual=False,
                 d_k=None, 
                 d_v=None,
                 d_ff=256, 
                 norm='BatchNorm', 
                 attn_dropout=0., 
                 dropout=0.,
                 head_dropout=0., 
                 act="gelu",
                 pe='zeros',
                 learn_pe=True,
                 pre_norm=False, 
                 res_attention=False, 
                 store_attn=False):
        super().__init__()
        
        
        self.seq_len = seq_len 
        self.patch_len = patch_len
        self.feat_dim = feat_dim
        
        self.individual = individual
        self.head_dropout = head_dropout
        # Patching

        assert not seq_len%patch_len, f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})"
        patch_num = seq_len //patch_len
        self.head_nf = d_model * patch_num
        self.patch_num = patch_num
        q_len = patch_num
        
        # Input encoding
        self.W_P = nn.Linear(patch_len,d_model)        # Projection of input random noise 
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model) #shape = (q_len,d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout) 
        
        # Transformer encoder (Blocks + Upsample)
        
        for i in range(depth):
            ## Blocks
            
            self.add_module(f'block{i}', TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn))
            ## Upsample
            self.add_module(f'upsample{i}', UpsampleAndReduce(d_model,scale_factor))
            q_len = q_len*scale_factor
            d_model = d_model//scale_factor
            assert not d_model%scale_factor, f"d_model ({d_model}) must be divisible by scale_factor ({scale_factor})"
            d_ff = 2*d_model
            
        self.encoder = nn.ModuleList([self.block0,self.upsample0,self.block1,self.upsample1,self.block2,self.upsample2])        
        self.fltten = Flatten_Head(individual=self.individual, n_vars=self.feat_dim, nf=self.head_nf, target_window=pred_len, head_dropout=self.head_dropout)

        
    def forward(self, x):   
        # x.shape : [BATCH_SIZE, FEATURE_DIM, seq_len]            
        b, f, s = x.shape
        # x.shape :[BATCH_SIZE, FEATURE_DIM, seq_len] -> [BATCH_SIZE, FEATURE_DIM, patch_num,patch_len]
        x = x.unfold(dimension=-1,size=self.patch_len,step=self.patch_len) 
        # u.shape : [BATCH_SIZE, FEATURE_DIM, patch_num, d_model]
        u = self.W_P(x)  
        # u.shape = [BATCH_SIZE, FEATURE_DIM, patch_num, d_model] -> [BATCH_SIZE*FEATURE_DIM, patch_num, d_model]            
        u = u.reshape(u.shape[0]*u.shape[1],u.shape[2],u.shape[3])  
        # x.shape : [BATCH_SIZE*FEATURE_DIM, patch_num, d_model] 
        x = u + self.W_pos               
        x = self.dropout(x)
        
        #outputs(x):[BATCH_SIZE*FEATURE_DIM,patch_num*(scale_factor)**depth, d_model//(scale_factor)**depth]            
        for layer in self.encoder:
            x = layer(x)    
                       
        #x:[BATCH_SIZE*FEATURE_DIM,patch_num*(scale_factor)**depth, d_model//(scale_factor)**depth] ->[bz,feat_dim,patch_num*(scale_factor)**depth,d_model//(scale_factor)**depth]
        out_s = x.shape[2]
        x = x.reshape(b,f,out_s,-1)
        #x:[bz,feat_dim,patch_num*(scale_factor)**depth,d_model//(scale_factor)**depth] -> [bz,feat_dim,patch_num*(scale_factor)**depth,d_model//(scale_factor)**depth]
        x = x.permute(0,1,3,2)
        #x:[bz,patch_num*(scale_factor)**depth,d_model//(scale_factor)**depth,feat_dim] -> [bz,feat_dim,pred_len]
        x = self.fltten(x)                                  
        return x    
         