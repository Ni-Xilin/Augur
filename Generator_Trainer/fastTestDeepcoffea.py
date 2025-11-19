import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import pathlib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import pickle

result_fpath = pathlib.Path("target_model/deepcoffea/code/Gcorrmatrix.npz")
loaded = np.load(result_fpath,allow_pickle=True)
print(1)