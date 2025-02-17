import torch
import torch.nn as nn
from transformers import AutoTokenizer,BertLayer,BertConfig
from einops import rearrange

from AbAgKer.taming.modules.autoencoder.qformer import Q_former
from AbAgKer.taming.modules.autoencoder.decoder import M_decoder


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

def positional_encoding(seq_len, dim, theta=10000.0):
    """
    Generate relative positional encoding for a sequence.
    Args:
        seq_len (int): The length of the sequence.
        dim (int): The dimension of the embeddings.
        theta (float): The base for the positional encoding.
    Returns:
        torch.Tensor: The positional encoding tensor.
    """
    assert dim % 2 == 0, "Embedding dimension must be even"
    half_dim = dim // 2
    positions = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    dims = torch.arange(half_dim, dtype=torch.float).unsqueeze(0)
    angle_rates = 1.0 / (theta ** (dims / half_dim))
    angle_rads = positions * angle_rates

    pos_encoding = torch.zeros((seq_len, dim))
    pos_encoding[:, 0::2] = torch.sin(angle_rads)
    pos_encoding[:, 1::2] = torch.cos(angle_rads)

    return pos_encoding


class CELoss(nn.Module):
    def __init__(self,model_name,d_model,
                 kd_weight,kon_weight,koff_weight
                 ):
        super().__init__()
        ### param init 
        self.model_name = model_name
        self.gelu = nn.GELU()

        ### loss weights 
        self.kd_weight=kd_weight
        self.kon_weight=kon_weight
        self.koff_weight=koff_weight
        
        # 损失函数定义
        self.huber_loss = nn.HuberLoss(reduction='mean', delta=1.0)
        self.L1Loss = torch.nn.SmoothL1Loss(reduction='mean')

        # DT_pre layer
        self.kd_pre = nn.Linear(d_model, 1)         
        self.kon_pre = nn.Linear(d_model, 1) 
        self.koff_pre = nn.Linear(d_model, 1) 

    def DT_pre(self, outputs_pm):

        x = rearrange(outputs_pm , 'b l i -> b i l').contiguous()
        x = self.gelu(self.DT_pre_len(x))

        x = rearrange( x , 'b i l -> b l i').contiguous()
        kd_pre = self.kd_pre(x)
        kon_pre = self.kon_pre(x)
        koff_pre = self.koff_pre(x)

        output = torch.squeeze(output)

        return kd_pre, kon_pre, koff_pre

    def forward(self, inputs_pm, kd, kon, koff, optimizer_idx, split="train"): 

        if optimizer_idx==0:

            kd_pre, kon_pre, koff_pre = self.DT_pre(inputs_pm)

            # 这里同时计算所有的token进行损失计算
            loss_kd = self.L1Loss(kd_pre,kd)
            loss_kon = self.huber_loss(kon_pre,kon)
            loss_off = self.huber_loss(kd_pre,kd)
                
            loss = loss_kd * self.kd_weight + loss_off * self.koff_weight  

            log = {"{}/loss".format(split): loss.clone().detach(), 
                "{}/loss_kd".format(split): loss_kd.clone().detach(),
                "{}/loss_kon".format(split): loss_kon.clone().detach(),
                "{}/loss_koff".format(split): loss_kon.clone().detach()
                }

        return loss,log
    

class mixed_model(nn.Module):
    def __init__(self,model_name,
                 p_edim,m_edim,
                 d_model,num_heads,fd_dim,num_layers,dp_out
                 ):
        super().__init__()
        ### param init 
        self.model_name = model_name
        self.gelu = nn.GELU() 

        ### feature mixed module init--------------------------------------
        config = BertConfig(
                    hidden_size = d_model,
                    num_attention_heads = num_heads,
                    intermediate_size = fd_dim,
                    hidden_dropout_prob = dp_out
                )
        self.layers = nn.ModuleList([
            BertLayer(config)
            for _ in range(num_layers)
        ])

        # feature transfer layers
        self.h_emb = nn.Linear(p_edim, d_model)  
        self.l_emb = nn.Linear(m_edim, d_model)  
        self.x_emb = nn.Linear(m_edim, d_model)  
        self.norm_dmodel = nn.LayerNorm(d_model)


    def f_extr(self,h,modal=0.0):
        # 位置编码信息
        h_position = positional_encoding(h.shape[1], h.shape[2]).to(h.device)

        # 模态信息
        h_modal = torch.tensor([modal],dtype=torch.float32).to(h.device) # 蛋白质模态信息1；分子模态为0；

        # 信息注入
        a_new = self.norm_dmodel(h) + h_position + h_modal

        return a_new
    

    def forward(self, Ab_H_emb, Ab_L_emb, Ag_emb, training_mask):
        Ab_H_emb = self.h_emb(Ab_H_emb)
        Ab_L_emb = self.l_emb(Ab_L_emb)
        Ag_emb = self.x_emb(Ag_emb)

        # 特征维度变换与拼接
        Ab_H = self.gelu(self.f_transfer(Ab_H_emb,0.0)) # [b, 896, 768]
        Ab_L = self.gelu(self.f_transfer(Ab_L_emb,1.0)) # [b, 896, 768]
        Ag = self.gelu(self.f_transfer(Ag_emb,2.0)) # [b , 128, 768]
     
        inputs_pm = torch.cat([Ab_H, Ab_L,Ag],dim=2)

        # mixed model process
        for layer in self.layers:
            layer_output = layer(inputs_pm,None)
            inputs_pm = layer_output[0] 
        
        return inputs_pm