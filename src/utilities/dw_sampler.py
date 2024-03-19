#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:57:31 2020

Distributed Weighted Sampler. 
Thanks to and with assistance from @ptrblk, DL Software engineer, NVIDIA
https://discuss.pytorch.org/u/ptrblck/summary

Not currently used in model training for paper. 

@author: mlim-user
"""
import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True, shuffle=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle
        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        self.targets = self.dataset.episode_wise_targets
        self.targets = self.targets[0:self.total_size:self.num_replicas]


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]

        assert len(indices) == self.num_samples

        assert len(self.targets) == self.num_samples
        weights = self.calculate_weights(self.targets)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch