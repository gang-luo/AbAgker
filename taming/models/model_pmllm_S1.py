import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel,AutoModelForMaskedLM,AutoTokenizer

from PALM.Code.bert_data_prepare.tokenizer import get_tokenizer
from transformers import AutoTokenizer,AutoModel
from transformers import AutoModel,RoFormerModel

import torchmetrics
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef, MeanSquaredError, MeanAbsoluteError
from scipy.stats import pearsonr

from main import instantiate_from_config
from PALM.Code.bert_data_prepare.tokenizer import CommonTokenizer


class PmllmModel(pl.LightningModule): 
    def __init__(self,
                 mixed_config,
                 lossconfig,
                 model_name="pretrain",
                 ### llm config
                 frozen_H=False,
                 frozen_L=False,
                 frozen_esm2=False,
                 H_llm="pretrain_model/Heavy_roformer",
                 L_llm="pretrain_model/Light_roformer",
                 ems2_llm="pretrain_model/esm2_t30",
                 ab_maxlen = 512, #  预训练模型max len为512
                 ag_maxlen = 640, # agen_seqs max_len is 613
                 ### scheduler config
                 learning_rate=None,
                 scheduler_type=None,
                 ### others
                 training_evaluate=True,
                 ckpt_path=None,
                 ignore_keys=[]
                 ):
        super().__init__()
        self.model_name=model_name
        self.scheduler_type = scheduler_type
        
        ### cite the mixed-model and loss params
        self.ab_maxlen = ab_maxlen
        self.ag_maxlen = ag_maxlen

        # LLm training flag
        self.frozen_HRoformer_llm = frozen_H
        self.frozen_LRoformer_llm = frozen_L
        self.frozen_esm2_llm = frozen_esm2

        # arctecture and loss
        self.mixed_model=instantiate_from_config(mixed_config)
        self.loss = instantiate_from_config(lossconfig)
        
        # proteins and mols pretrained model       
        self.HeavyModel = AutoModel.from_pretrained(H_llm, output_hidden_states=True, return_dict=True)
        self.LightModel = AutoModel.from_pretrained(L_llm, output_hidden_states=True, return_dict=True)
        self.AntigenModel = AutoModel.from_pretrained(ems2_llm, output_hidden_states=True, return_dict=True)

        ### 重链 and 轻链 tokenizer
        self.tokenizer = CommonTokenizer(logger=None, add_hyphen=False)
        self.heavy_tokenizer = self.tokenizer.get_bert_tokenizer(max_len=self.ab_maxlen, tokenizer_dir=H_llm)
        # self.light_tokenizer = self.tokenizer.get_bert_tokenizer(max_len=self.ab_maxlen, tokenizer_dir=L_llm)
        self.antigen_tokenizer = AutoTokenizer.from_pretrained(ems2_llm,trust_remote_code=True,max_len = self.ag_maxlen)

        # some funtional config
        self.learning_rate = learning_rate
        self.training_evaluate=training_evaluate
        self.training_mask = 0 

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        ### 评估策略
        # 初始化评估指标
        self.pearson = PearsonCorrCoef()
        self.spearman = SpearmanCorrCoef()
        self.pearson2 = PearsonCorrCoef()
        self.spearman2 = SpearmanCorrCoef()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}") 


    def H_encoder(self, inputs):    
        outputs= self.HeavyModel(**inputs) 
        H_outputs = outputs.last_hidden_state
        return H_outputs 

    def L_encoder(self, inputs):
        outputs = self.LightModel(**inputs) 
        L_outputs = outputs.last_hidden_state
        return L_outputs
    
    def Ag_encoder(self, inputs):
        outputs = self.AntigenModel(**inputs) 
        G_outputs = outputs.last_hidden_state
        return G_outputs

    def forward(self, H_seq,L_seq,X_seq, training_mask):        
        # token化
        H_info = self.heavy_tokenizer(H_seq, padding="max_length", max_length=self.ab_maxlen, truncation=True, return_tensors="pt").to(self.HeavyModel.device)
        L_info = self.heavy_tokenizer(L_seq, padding="max_length", max_length=self.ab_maxlen, truncation=True, return_tensors="pt").to(self.HeavyModel.device)
        X_info = self.antigen_tokenizer(X_seq, 
                                     max_length = self.ag_maxlen, 
                                     padding="max_length",
                                     truncation=True, 
                                     return_tensors="pt",
                                     add_special_tokens=True).to(self.AntigenModel.device)


        # tensor化
        Ab_H_emb = self.H_encoder(H_info)
        Ab_L_emb = self.L_encoder(L_info)
        Ag_emb = self.Ag_encoder(X_info)

        # 特征融合，预测DTA 
        outputs_pm = self.mixed_model(Ab_H_emb, Ab_L_emb, Ag_emb, training_mask)

        return outputs_pm

    def validation_step(self, batch, batch_idx):
        
        # 验证阶段
        Ab_H, Ab_L, Ag_X, kd, kon, koff = self.get_input(batch)
        outputs_pm = self(Ab_H,Ab_L,Ag_X, self.training_mask)

        total_loss, log_dict = self.loss(outputs_pm, kd, kon, koff, 0, split="val") 
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        kd_pre, _, koff_pre = self.loss.DT_pre(outputs_pm)

        # 更新评估指标
        self.pearson.update(kd_pre, kd)
        self.spearman.update(kd_pre, kd)
        self.pearson2.update(koff_pre, koff)
        self.spearman2.update(koff_pre, koff)

    def validation_epoch_end(self, outputs):
        # 计算每个指标的值
        pearson_r = self.pearson.compute().item()
        spearman_r = self.spearman.compute().item() 
        self.log('val/kd_pearson', pearson_r, prog_bar=False)
        self.log('val/kd_spearman', spearman_r, prog_bar=False)
        self.pearson.reset()
        self.spearman.reset()
        
        pearson_r2 = self.pearson2.compute().item()
        spearman_r2 = self.spearman2.compute().item() 
        self.log('val/koff_pearson', pearson_r2, prog_bar=False)
        self.log('val/koff_spearman', spearman_r2, prog_bar=False)
        self.pearson2.reset()
        self.spearman2.reset()


    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if self.model_name == "AbAg":

            Ab_H, Ab_L, Ag_X, kd, kon, koff = self.get_input(batch)
            outputs_pm = self(Ab_H,Ab_L,Ag_X, self.training_mask)

        # elif self.model_name == "PP": 

        #     prot1, prot2, kd, kon, koff = self.get_input(batch)
        #     outputs_pm = self(prot1,prot2)

        # loss
        if optimizer_idx == 0: 

            total_loss, log_dict = self.loss(outputs_pm, kd, kon, koff, optimizer_idx,split="train") 
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            return total_loss


    def get_input(self, batch):

        pdb_name = batch["pdb"]
        seqs = batch["seqs"]
        kd = batch["kd"].to(torch.float32)
        kon = batch["kon"].to(torch.float32)  
        koff = batch["koff"].to(torch.float32)

        # kon =[-i for i in batch["kon"]] # 转为正值辅助损失计算
        Ab_H = [" ".join(self.tokenizer.split(i)) for i in seqs["H"]]
        Ab_L = [" ".join(self.tokenizer.split(i)) for i in seqs["L"]]
        Ag_X = seqs["X"]
        

        return Ab_H, Ab_L, Ag_X, kd, kon, koff

    def configure_optimizers(self):
        if self.frozen_HRoformer_llm:
            for param in self.HeavyModel.parameters():
                param.requires_grad = False
        if self.frozen_LRoformer_llm:
            for param in self.LightModel.parameters():
                param.requires_grad = False
        if self.frozen_esm2_llm:
            for param in self.AntigenModel.parameters():
                param.requires_grad = False

        lr = self.learning_rate
        opt_mixed = torch.optim.AdamW( list(self.mixed_model.parameters()) +
                                        list(self.loss.parameters()), 
                                        lr=lr, betas=(0.5, 0.9))

        if self.scheduler_type == "None":
            return {"optimizer": opt_mixed}
