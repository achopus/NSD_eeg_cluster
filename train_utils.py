import os
import torch
import numpy as np
from torch.nn import Module
from convnext import ConvNeXt1D
from copy import deepcopy

class TrainSaveObject:
    def __init__(self, epoch: int, model: Module, lr: float,
                 L_emb: list[float], L_cls: list[float], L_dmn: list[float],
                 L_all: list[float], ACC: list[float],
                 model_config: dict[str, int]):
        
        self.epoch = epoch
        self.model = deepcopy(model.cpu())
        self.lr = lr
        self.L_emb = L_emb
        self.L_clc = L_cls
        self.L_dmn = L_dmn
        self.L_all = L_all
        self.ACC = ACC
        self.model_config = model_config
        
    def save(self, save_folder: str) -> None:
        torch.save(self, os.path.join(save_folder, f"train_progress_{self.epoch}.pth"))
        
    
    def __str__(self) -> str:
        out_str = f"Epoch: {self.epoch} | lr: {self.lr:.5f}\n"
        LA = np.mean(self.L_all)
        LE = np.mean(self.L_emb)
        LC = np.mean(self.L_clc)
        LD = np.mean(self.L_dmn)
        ACC = np.mean(self.ACC)
        out_str += f"Loss total: {LA:.3f}\n"
        out_str += f"Loss E:{LE:.3f} | C:{LC:.3f} | D:{LD:.3f}\n"
        out_str += f"Accuracy: {ACC:.2f} %"
        return out_str