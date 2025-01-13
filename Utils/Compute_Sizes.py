# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:15:10 2021
@author: jpeeples
"""
from lightning import LightningDataModule


def get_feat_size_lightning(
    data_module: LightningDataModule, model=None, audio_feature_extractor=None
):

    # Get single example of data to compute shape (use CPU instead of GPU)
    inputs, labels, index = next(iter(data_module.train_dataloader()))
    ip = inputs[0].unsqueeze(0)

    feats = audio_feature_extractor(ip)
    print("Audio Feature Shape: ", ip.shape)
    if model:
        feats = model(feats)

    # Compute out size
    out_size = feats.flatten(1).shape[-1]

    return out_size
