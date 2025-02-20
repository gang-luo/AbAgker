import torch
import torch.nn as nn
from transformers import AutoTokenizer,BertLayer,BertConfig
from einops import rearrange

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
    def __init__(self,d_model,
                 kd_weight,kon_weight,koff_weight
                 ):
        super().__init__()
        ### loss weights 
        self.kd_weight=kd_weight
        self.kon_weight=kon_weight
        self.koff_weight=koff_weight
        
        # 损失函数定义
        self.gelu = nn.GELU()
        self.huber_loss = nn.HuberLoss(reduction='mean', delta=1.0)
        self.L1Loss = torch.nn.SmoothL1Loss(reduction='mean')

        # DT layer 
        self.mid = 256
        self.DT_pre_len1 = nn.Linear( 1664, d_model)
        self.DT_pre_len2 = nn.Linear(d_model, self.mid)
        self.DT_pre_len3 = nn.Linear(self.mid, 1)
        self.DT_pre_len4 = nn.Linear(d_model, self.mid)
        
        # DT_pre layer
        self.kd_pre = nn.Linear(self.mid, 1)         
        self.kon_pre = nn.Linear(self.mid, 1) 
        self.koff_pre = nn.Linear(self.mid, 1) 

    def DT_pre(self, outputs_pm): 

        # outputs_pm dim [b, 1024+640, 768]

        x = rearrange(outputs_pm , 'b l i -> b i l').contiguous()
        x = self.gelu(self.DT_pre_len1(x))
        x = self.gelu(self.DT_pre_len2(x))
        outputs_feature = self.gelu(self.DT_pre_len3(x))

        x = rearrange(outputs_feature , 'b i l -> b l i').contiguous()
        outputs_pre =  self.gelu(self.DT_pre_len4(x))

        kd_pre = self.kd_pre(outputs_pre)
        kon_pre = self.kon_pre(outputs_pre)
        koff_pre = self.koff_pre(outputs_pre)

        return kd_pre.squeeze(), kon_pre.squeeze(), koff_pre.squeeze()

    def forward(self, inputs_pm, kd, kon, koff, optimizer_idx, split="train"): 

        if optimizer_idx==0:

            kd_pre, kon_pre, koff_pre = self.DT_pre(inputs_pm)

            # 这里同时计算所有的token进行损失计算
            loss_kd = self.L1Loss(kd_pre, kd)
            loss_off = self.huber_loss(koff_pre, koff) 
            loss_kon = self.huber_loss((koff_pre-kd_pre), kon) # loss_kon = self.huber_loss((kon_pre), kon)
                
            loss = loss_kd * self.kd_weight + loss_off * self.koff_weight  

            log = {"{}/loss".format(split): loss.clone().detach(), 
                "{}/loss_kd".format(split): loss_kd.clone().detach(),
                "{}/loss_kon".format(split): loss_kon.clone().detach(),
                "{}/loss_koff".format(split): loss_off.clone().detach()
                }

        return loss,log
    

class mixed_model(nn.Module):
    def __init__(self,model_name,
                 ab_edim,ag_edim,
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
        # self.h_emb = nn.Linear(ab_edim, d_model)  
        # self.l_emb = nn.Linear(ab_edim, d_model)  
        self.x_emb = nn.Linear(ag_edim, d_model)  
        self.norm_dmodel = nn.LayerNorm(d_model)


    def f_extr(self,h,modal=0.0):
        # 位置编码信息
        h_position = positional_encoding(h.shape[1], h.shape[2]).to(h.device)

        # 模态信息
        h_modal = torch.tensor([modal],dtype=torch.float32).to(h.device) # 蛋白质模态信息1；分子模态为0；

        # 信息注入
        h_new = self.norm_dmodel(h) + h_position + h_modal

        return h_new
    

    def forward(self, Ab_H_emb, Ab_L_emb, Ag_emb, training_mask):
        # 特征维度变换
        # Ab_H_emb = self.h_emb(Ab_H_emb)
        # Ab_L_emb = self.l_emb(Ab_L_emb)
        Ag_emb = self.x_emb(Ag_emb)

        # 模态信息注入
        Ab_H = self.gelu(self.f_extr(Ab_H_emb,0.0)) # [b, 512, 768]
        Ab_L = self.gelu(self.f_extr(Ab_L_emb,1.0)) # [b, 512, 768]
        Ag = self.gelu(self.f_extr(Ag_emb,2.0)) # [b , 640, 768]

        # 特征拼接
        inputs_pm = torch.cat([Ab_H, Ab_L,Ag],dim=1)

        # 特征混合
        for layer in self.layers:
            layer_output = layer(inputs_pm,None)
            inputs_pm = layer_output[0] 
        
        return inputs_pm