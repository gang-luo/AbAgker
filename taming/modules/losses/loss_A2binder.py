import torch
import torch.nn as nn
from transformers import AutoTokenizer,BertLayer,BertConfig
from einops import rearrange

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

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
        self.MSEloss = torch.nn.MSELoss(reduction='mean')
        self.MAEloss = torch.nn.L1Loss(reduction='mean')

    def forward(self, kd_pre, koff_pre, kd, kon, koff, optimizer_idx, split="train"): 

        if optimizer_idx==0:
            
            # 这里同时计算所有的token进行损失计算
            loss_kd = self.MSEloss(kd_pre, kd)
            loss_off = self.MSEloss(koff_pre, koff) 
            loss_kon = self.MSEloss((koff_pre-kd_pre), kon) # loss_kon = self.huber_loss((kon_pre), kon)
                
            loss = loss_kd * self.kd_weight + loss_off * self.koff_weight  

            log = {"{}/loss".format(split): loss.clone().detach(), 
                "{}/loss_kd".format(split): loss_kd.clone().detach(),
                "{}/loss_kon".format(split): loss_kon.clone().detach(),
                "{}/loss_koff".format(split): loss_off.clone().detach(),
                "{}/loss_kd_L1".format(split): self.MAEloss(kd_pre, kd).clone().detach(),
                "{}/loss_koff_L1".format(split): self.MAEloss(koff_pre, koff).clone().detach()
                }

        return loss,log
    
