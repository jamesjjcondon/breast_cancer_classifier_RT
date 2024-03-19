#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 21:37:05 2021

@author: drjc
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class DataIter(Dataset):
    def __init__(self):
        self.data_np = np.array([x for x in range(24000000)])
        self.data = [x for x in range(24000000)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data_np[idx]
        data = np.array([data], dtype=np.int64)
        return torch.tensor(data)


train_data = DataIter()
train_loader = DataLoader(train_data, batch_size=5,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=False,
                          num_workers=2)

for i, item in enumerate(train_loader):
    if i % 1000 == 0:
        print(i)
