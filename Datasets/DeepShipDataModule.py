#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:15:42 2025
Code modified from: https://github.com/Advanced-Vision-and-Learning-Lab/HLTDNN/blob/master/Datasets/DeepShipDataModule.py
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import pytorch_lightning as pl
import pdb
from Utils.Get_min_max import get_min_max_minibatch

from Datasets.Get_preprocessed_data import process_data

class DeepShipSegments(Dataset):
    def __init__(self, parent_folder, train_split=.7,val_test_split=2/3,
                 partition='train', random_seed= 42, shuffle = False, transform=None, 
                 target_transform=None):
        self.parent_folder = parent_folder
        self.folder_lists = {
            'train': [],
            'test': [],
            'val': []
        }
        self.train_split = train_split
        self.val_test_split = val_test_split
        self.partition = partition
        self.transform = transform
        self.shuffle = shuffle
        self.target_transform = target_transform
        self.random_seed = random_seed
        self.class_mapping = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}
        self.normalize = False 
        self.overall_min = None  
        self.overall_max = None  

        # Loop over each label and subfolder
        for label in ['Cargo', 'Passengership', 'Tanker', 'Tug']:
            label_path = os.path.join(parent_folder, label)
            subfolders = os.listdir(label_path)
            
            # Split subfolders into training, testing, and validation sets
            subfolders_train, subfolders_test_val = train_test_split(subfolders, 
                                                                     train_size=train_split, 
                                                                     shuffle=self.shuffle, 
                                                                     random_state=self.random_seed)
            subfolders_test, subfolders_val = train_test_split(subfolders_test_val, 
                                                               train_size=self.val_test_split, 
                                                               shuffle=self.shuffle, 
                                                               random_state=self.random_seed)

            # Add subfolders to appropriate folder list
            for subfolder in subfolders_train:
                subfolder_path = os.path.join(label_path, subfolder)
                self.folder_lists['train'].append((subfolder_path, self.class_mapping[label]))

            for subfolder in subfolders_test:
                subfolder_path = os.path.join(label_path, subfolder)
                self.folder_lists['test'].append((subfolder_path, self.class_mapping[label]))

            for subfolder in subfolders_val:
                subfolder_path = os.path.join(label_path, subfolder)
                self.folder_lists['val'].append((subfolder_path, self.class_mapping[label]))

        self.segment_lists = {
            'train': [],
            'test': [],
            'val': []
        }

        # Loop over each folder list and add corresponding files to file list
        for split in ['train', 'test', 'val']:
            for folder in self.folder_lists[split]:
                for root, dirs, files in os.walk(folder[0]):
                    for file in files:
                        if file.endswith('.wav'):
                            file_path = os.path.join(root, file)
                            label = folder[1]
                            self.segment_lists[split].append((file_path, label))


    def __len__(self):
        return len(self.segment_lists[self.partition])

    def set_normalization(self, overall_min, overall_max):
        self.overall_min = overall_min
        self.overall_max = overall_max
        self.normalize = True  # Enable normalization
    

    def __getitem__(self, idx):
    
        file_path, label = self.segment_lists[self.partition][idx]    
        try:
            sr, signal = wavfile.read(file_path, mmap=False)
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {e}")

        signal = signal.astype(np.float32)
        # Normalize the signal if normalization parameters are set
        if self.normalize:
            signal = 2 * (signal - self.overall_min) / (self.overall_max - self.overall_min) - 1

        label = torch.tensor(label)
        if self.target_transform:
            label = self.target_transform(label)
        return signal, label, idx
    

class DeepShipDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, pin_memory,sample_rate,segment_length, 
                 train_split=0.7, val_split=0.10, test_split=0.20, random_seed=42, shuffle=False):
        super().__init__()
        print(f"train_split: {train_split}, val_split: {val_split}, test_split: {test_split}")
        assert train_split + val_split + test_split == 1, "Splits must add up to 1"
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.norm_function = None
        self.global_min = None
        self.global_max = None
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.train_min, self.train_max= None, None
    def prepare_data(self):
        # Download or prepare data if necessary
        process_data(sample_rate=self.sample_rate, segment_length=self.segment_length)
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DeepShipSegments(self.data_dir, partition='train', 
                                                  random_seed=self.random_seed, 
                                                  shuffle=self.shuffle)
            self.val_dataset = DeepShipSegments(self.data_dir, partition='val', 
                                                random_seed=self.random_seed, 
                                                shuffle=self.shuffle)
            #Compute normalization parameters using only the training data subset
            self.train_min, self.train_max = get_min_max_minibatch(self.train_dataset, batch_size=self.batch_size['train']) 
            #Set normalization parameters on the full dataset
            self.train_dataset.set_normalization(self.train_min, self.train_max)
            self.val_dataset.set_normalization(self.train_min, self.train_max)


        if stage == 'test' or stage is None:
            self.test_dataset = DeepShipSegments(self.data_dir, partition='test', 
                                                random_seed=self.random_seed, 
                                                shuffle=self.shuffle)

            self.test_dataset.set_normalization(self.train_min, self.train_max)



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size['train'], 
                          shuffle=True, num_workers=self.num_workers, 
                          pin_memory=self.pin_memory, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size['val'], 
                          shuffle=False, num_workers=self.num_workers, 
                          pin_memory=self.pin_memory, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size['test'], 
                          shuffle=False, num_workers=self.num_workers, 
                          pin_memory=self.pin_memory)