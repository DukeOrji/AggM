# config.py

import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

SAVE = True
batch_size = 64

# Training
num_epochs = 50
weight_decay = 1e-4
momentum = 0.9

# Head LR
base_lr = 1e-2
min_lr = 1e-4

# Backbone LR
backbone_base_lr = 1e-3
backbone_min_lr = 1e-5