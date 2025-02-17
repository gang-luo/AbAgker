import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel,AutoModelForMaskedLM,AutoTokenizer

from PALM.Code.bert_data_prepare.tokenizer import get_tokenizer
from transformers import AutoTokenizer,AutoModel
from transformers import AutoModel,RoFormerModel

from main import instantiate_from_config

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

                 p_padding=1, 
                 m_padding=2,
                 p_mask=32,
                 m_mask=3,
                 ### others
                 training_evaluate=True,
                 ckpt_path=None,
                 ignore_keys=[],
                 ### scheduler config
                 learning_rate=None,
                 scheduler_type=None
                 ):
        super().__init__()
        self.scheduler_type = scheduler_type
        self.model_name=model_name
        self.m_embeding_layer = self.m_model.get_input_embeddings()
        
        # self.p_len = mixed_config.params.p_len
        # self.m_len = mixed_config.params.m_len
        # self.pocket_len = lossconfig.params.pocket_len

        # # tokenizer speice define 
        # self.p_padding=p_padding
        # self.p_mask=p_mask
        # self.m_padding=m_padding
        # self.m_mask=m_mask

        # LLm training flag
        self.frozen_HRoformer_llm = frozen_H
        self.frozen_LRoformer_llm = frozen_L
        self.frozen_esm2_llm = frozen_esm2

        # arctecture and loss
        self.mixed_model=instantiate_from_config(mixed_config)
        self.loss = instantiate_from_config(lossconfig)
        
        # proteins and mols pretrained model       
        self.H_Roformer = AutoModel.from_pretrained(H_llm, output_hidden_states=True, return_dict=True)
        self.L_Roformer = AutoModel.from_pretrained(L_llm, output_hidden_states=True, return_dict=True)
        self.esm2_t30 = AutoModel.from_pretrained(ems2_llm, deterministic_eval=True, trust_remote_code=True)

        ### 重链 and 轻链 tokenizer
        self.tokenizer = get_tokenizer(tokenizer_name="common",add_hyphen=False,logger=None,vocab_dir="PALM/ProcessedData/vocab/heavy-2-3.csv")
        self.H_tokenizer = self.tokenizer.get_bert_tokenizer(max_len=256, tokenizer_dir = H_llm)
        self.L_tokenizer  = self.tokenizer.get_bert_tokenizer(max_len=256, tokenizer_dir = L_llm)
        self.agen_tokenizer = AutoTokenizer.from_pretrained(ems2_llm, trust_remote_code=True)    

        # some funtional config
        self.learning_rate = learning_rate
        self.training_evaluate=training_evaluate
        self.training_mask = 0 
        self.flag = 0 

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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
        outputs= self.H_Roformer(**inputs) 
        H_outputs = outputs.last_hidden_state
        return H_outputs 

    def L_encoder(self, inputs):
        outputs = self.L_Roformer(**inputs) 
        L_outputs = outputs.hidden_states[-1] 
        return L_outputs
    
    def Agen_encoder(self, inputs):
        outputs = self.esm2_t30(**inputs) 
        agen_outputs = outputs.hidden_states[-1] 
        return agen_outputs

    def forward(self, H_seq,L_seq,X_seq, training_mask):        
        # token化
        H_info = self.H_tokenizer(" ".join(self.tokenizer.split(H_seq)),
                                                    padding="max_length",
                                                    max_length=256,
                                                    truncation=True,
                                                    return_tensors="pt")
        L_info = self.L_tokenizer(" ".join(self.tokenizer.split(L_seq)),
                                                    padding="max_length",
                                                    max_length=256,
                                                    truncation=True,
                                                    return_tensors="pt")
        X_info = self.agen_tokenizer(X_seq, max_length = 512, padding="max_length",truncation=True, return_tensors="pt",add_special_tokens=True)

        # tensor化
        Ab_H_emb = self.H_encoder(H_info)
        Ab_L_emb = self.L_encoder(L_info)
        Ag_emb = self.Agen_encoder(X_info)

        # 特征融合，预测DTA 
        _, kd_pre, _, koff_pre = self.mixed_model(Ab_H_emb, Ab_L_emb, Ag_emb, training_mask)

        return kd_pre, koff_pre 


    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if self.model_name == "AbAg":

            Ab_H, Ab_L, Ag_X, kd, kon, koff = self.get_input(batch)
            outputs_pm = self(Ab_H,Ab_L,Ag_X)

        elif self.model_name == "PP": 

            prot1, prot2, kd, kon, koff = self.get_input(batch)
            outputs_pm = self(prot1,prot2)

        # loss
        if optimizer_idx == 0: 

            total_loss, log_dict = self.loss(outputs_pm, kd, kon, koff, optimizer_idx,split="train") 
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            return total_loss

    ### data read -----------------------------------------------------------------------------
    def get_input(self, batch):

        pdb_name = batch["pdb"]
        seqs = batch["seqs"]
        kd = batch["kd"]   
        kon = batch["kon"]   
        koff = batch["koff"]  
        # kon =[-i for i in batch["kon"]] # 转为正值辅助损失计算

        if self.model_name == "AbAg":
            Ab_H = self.HL_tokenizer(seqs["H"], max_length=self.p_len, padding="max_length",truncation=True,return_tensors="pt", add_special_tokens=True).to(self.p_model.device)
            Ab_L = self.HL_tokenizer(seqs["L"], max_length=self.p_len, padding="max_length",truncation=True,return_tensors="pt", add_special_tokens=True).to(self.p_model.device)
            Ag_X = self.agen_tokenizer(seqs["X"], max_length=self.m_len, padding="max_length",truncation=True, return_tensors="pt",add_special_tokens=True).to(self.p_model.device)
            return Ab_H, Ab_L, Ag_X, kd, kon, koff
        
        elif self.model_name == "PP": 
            prot1 = self.HL_tokenizer(seqs["P1"], max_length=self.p_len, padding="max_length",truncation=True,return_tensors="pt", add_special_tokens=True).to(self.p_model.device)
            prot2 = self.HL_tokenizer(seqs["P2"], max_length=self.p_len, padding="max_length",truncation=True,return_tensors="pt", add_special_tokens=True).to(self.p_model.device)
            
            return  prot1, prot2, kd, kon, koff

    def configure_optimizers(self):
        if self.frozen_HRoformer_llm:
            for param in self.H_Roformer.parameters():
                param.requires_grad = False
        if self.frozen_LRoformer_llm:
            for param in self.L_Roformer.parameters():
                param.requires_grad = False
        if self.frozen_esm2_llm:
            for param in self.esm2_t30.parameters():
                param.requires_grad = False


        lr = self.learning_rate
        opt_mixed = torch.optim.AdamW(
                                        list(self.mixed_model.parameters()) +
                                        list(self.loss.parameters()), 
                                        lr=lr, betas=(0.5, 0.9))

        if self.scheduler_type == "None":
            return {"optimizer": opt_mixed}


    # def validation_step(self, batch, batch_idx):
        
    #     p, m, t, pocket = self.get_input(batch)

    #     # 关于分子生成部分的评估
    #     outputs_pm_task1, m_task1, y_pocket_task1, _, t_loss_task1 = self(p,m,t,pocket,0)
    #     _ , log_dict_task1 = self.loss(outputs_pm_task1, m_task1, y_pocket_task1, t_loss_task1, self.m_embeding_layer, 0, 0, split="val")  # loss评估计算
    #     self.log_dict(log_dict_task1, prog_bar=False, logger=True, on_step=True, on_epoch=True) 
    #     self._evaluate_pocket("val", outputs_pm_task1, y_pocket_task1, m_task1["input_ids"], 0)

    #     # 关于DTA分数预测的评估
    #     outputs_pm_task2, m_task2, y_pocket_task2, _, t_loss_task2 = self(p,m,t,pocket,1)
    #     _ , log_dict_task2 = self.loss(outputs_pm_task2, m_task2, y_pocket_task2, t_loss_task2, self.m_embeding_layer, 0, 1, split="val")  # DTA评估计算
    #     self.log_dict(log_dict_task2, prog_bar=False, logger=True, on_step=True, on_epoch=True) 
    #     self._evaluate_pocket("val", outputs_pm_task2, y_pocket_task2, m_task2["input_ids"], 1)
       
