import os
import numpy as np
from torch.utils.data import Dataset
import json
import torch
import random

class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__() 
        # self.data = None
        self.data = []
        self.data_seed=56

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class DataTrain(CustomBase):
    def __init__(self, training_list_file):
        super().__init__()

        random.seed(self.data_seed)
        with open(training_list_file, "r") as f:
            self.data = json.load(f)
        random.shuffle(self.data)

        
class DataVal(CustomBase):
    def __init__(self, val_list_file):
        super().__init__()

        random.seed(self.data_seed)
        with open(val_list_file, "r") as f:
            self.data = json.load(f)
        random.shuffle(self.data)


# class DataTrain(CustomBase):
#     def __init__(self, max_lens_protein,max_lens_mols,training_list_file):
#         super().__init__()
        
#          # 加载JSON数据文件
#         with open(training_list_file, "r") as f:
#             # self.data = json.load(f)
#             json_info = json.load(f)
#             if json_info[0]["text"]:
#                 for item in json_info:
#                     protein = item["protein"]
#                     mol = item["mol"]
#                     text = item["text"]
                    
#                     if len(protein)<=max_lens_protein and len(mol)<=max_lens_mols:
#                     # 将数据追加到列表中
#                         self.data.append({
#                             "protein": protein,
#                             "mol": mol,
#                             "text": text
#                         })
#             else:
#                 for item in json_info:
#                     mol = item["mol"]
#                     protein = item["protein"]
                    
#                     if len(protein)<=max_lens_protein and len(mol)<=max_lens_mols:
#                     # 将数据追加到列表中
#                         self.data.append({
#                             "mol": mol,
#                             "protein": protein,
#                         })
        
# class DataVal(CustomBase):
#     def __init__(self, max_lens_protein,max_lens_mols, val_list_file):
#         super().__init__()
#          # 加载JSON数据文件
#         with open(val_list_file, "r") as f:
#             # self.data = json.load(f)
#             json_info = json.load(f)
#             for item in json_info:
#                 protein = item["protein"]
#                 mol = item["mol"]
#                 text = item["text"]
                
#                 if len(protein)<=max_lens_protein and len(mol)<=max_lens_mols:
#                 # 将数据追加到列表中
#                     self.data.append({
#                         "protein": protein,
#                         "mol": mol,
#                         "text": text
#                     })
        