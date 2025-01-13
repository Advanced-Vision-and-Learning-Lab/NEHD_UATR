#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:15:42 2025
Code modified from: https://github.com/Advanced-Vision-and-Learning-Lab/HLTDNN/blob/master/Datasets/Get_min_max.py
"""

import numpy as np


from torch.utils.data import DataLoader


def get_min_max_minibatch(dataset, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    min_values = []
    max_values = []

    for signals, _, _i in loader:
        min_values.append(signals.min().item())
        max_values.append(signals.max().item())

    overall_min = np.min(min_values)
    overall_max = np.max(max_values)
    return overall_min, overall_max
