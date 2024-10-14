import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_utils import TrainSaveObject

folder = "train_progress"
L = []
max_acc = 0
obj_best = None
files = os.listdir(folder)
for file in files:
    obj: TrainSaveObject = torch.load(os.path.join(folder, file))
    L.extend(obj.L_all)
    acc = np.mean(obj.ACC)
    if acc > max_acc:
        max_acc = acc
        obj_best = obj
        
k = 25
L_ma = np.convolve(L, np.ones(k) / k, mode="valid")
plt.figure()
plt.plot(L_ma)
it_per_ep = len(L) // len(files)
plt.vlines(np.linspace(it_per_ep, it_per_ep * len(files), len(files)), ymin=-1, ymax=100, color='gray', alpha=0.25)
plt.xlim(0, len(L_ma))
plt.ylim(min(L), max(L))
plt.show()