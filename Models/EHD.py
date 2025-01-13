# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:52:40 2020
Function to generate EHD histogram feature maps
@author: jpeeples, luke saleh
"""
import numpy as np
from scipy import signal,ndimage
import torch.nn.functional as F
import torch
import torch.types
import torch.nn as nn

    
class EHD(nn.Module):
    def __init__(self, in_channels, angle_res, normalize_kernel,
                 dilation, threshold, window_size, stride, normalize_count,
                aggregation_type, kernel_size= 3,device = 'cpu'):
        
        super(EHD, self).__init__()
        self.kernel_size = kernel_size
        self.angle_res = angle_res
        self.normalize_kernel = normalize_kernel
        self.in_channels = in_channels
    
        self.dilation = dilation
        self.threshold = threshold
        self.window_size = window_size
        self.window_scale = np.prod(np.asarray(self.window_size))
        self.stride = stride
        self.normalize_count = normalize_count
        self.aggregation_type = aggregation_type

        self.device = device
    
        
        
        #Generate masks based on parameters
        masks = EHD.Generate_masks(mask_size= self.kernel_size,
                            angle_res= self.angle_res,
                            normalize= self.normalize_kernel)
    
        #Convolve input with filters, expand masks to match input channels
        masks = torch.tensor(masks).float()
        masks = masks.unsqueeze(1)
        
        #Replicate masks along first dimension (for independently)
        masks = masks.repeat(in_channels,1,1,1)

        self.masks = masks
        # Call masks now that they are made
        self.num_orientations = self.masks.shape[0] // in_channels
    
    def forward(self,x):
        self.masks = self.masks.to(x.device)
        #Treat independently
        x = F.conv2d(x, self.masks,dilation= self.dilation, groups= self.in_channels)
        
        #Find max response
        [value,index] = torch.max(x,dim=1)
        
        #Set edge responses to "no edge" if not larger than threshold
        num_orientations = self.num_orientations
        index[value< self.threshold] = num_orientations
        
        feat_vect = []
        no_edge = []
        window_scale = self.window_scale
        
        
        #Aggregate EHD feature over each channel (TBD vectorize)
        for channel in range(0, self.in_channels):
            for edge in range(0,num_orientations):
                #Average count
                feat_vect.append((F.avg_pool2d((index==edge).unsqueeze(1).float(),
                                self.window_size ,stride= self.stride,
                                count_include_pad=False).squeeze(1)))
            
            #Compute no edge for each channel
            no_edge.append((F.avg_pool2d((index==edge+1).unsqueeze(1).float(),
                            self.window_size ,stride= self.stride,
                            count_include_pad=False).squeeze(1)))
        

        #Return vector, scale feature if desired
        if self.normalize_count:
            feat_vect = torch.stack(feat_vect,dim=1)
            no_edge = torch.stack(no_edge,dim=1)
        else:
            feat_vect = torch.stack(feat_vect,dim=1) * window_scale
            no_edge = torch.stack(no_edge,dim=1) * window_scale
            
        #For multichannel, need to rearrange feature to be in consistent order of NEHD
        #Eg., all no edge channels should be last feature maps
        feat_vect = torch.cat((feat_vect,no_edge),dim=1)
            
        #Only want histogram representations
        if (self.aggregation_type =='GAP'): 
            feat_vect = F.adaptive_avg_pool2d(feat_vect,1)

        
        return feat_vect
    
    @staticmethod
    def Generate_masks(mask_size=3,angle_res=45,normalize=False,rotate=False):
        
        #Make sure masks are appropiate size. Should not be less than 3x3 and needs
        #to be odd size
        if type(mask_size) is list:
            mask_size = mask_size[0]
        if mask_size < 3:
            mask_size = 3
        elif ((mask_size % 2) == 0):
            mask_size = mask_size + 1
        else:
            pass
        
        if mask_size == 3:
            if rotate:
                Gy = np.outer(np.array([1,2,1]).T,np.array([1,0,-1]))
            else:
                Gy = np.outer(np.array([1,0,-1]).T,np.array([1,2,1]))
        else:
            if rotate:
                Gy = np.outer(np.array([1,2,1]).T,np.array([1,0,-1]))
            else:
                Gy = np.outer(np.array([1,0,-1]).T,np.array([1,2,1]))
            dim = np.arange(5,mask_size+1,2)
            expand_mask =  np.outer(np.array([1,2,1]).T,np.array([1,2,1]))
            for size in dim:
                Gy = signal.convolve2d(expand_mask,Gy)
        
        #Generate horizontal masks
        angles = np.arange(0,360,angle_res)
        masks = np.zeros((len(angles),mask_size,mask_size))
        
        #TBD: improve for masks sizes larger than 
        for rot_angle in range(0,len(angles)):
            masks[rot_angle,:,:] = ndimage.rotate(Gy,angles[rot_angle],reshape=False,
                                                mode='nearest')
            
        
        #Normalize masks if desired
        if normalize:
            if mask_size == 3:
                masks = (1/8) * masks
            else:
                masks = (1/8) * (1/16)**len(dim) * masks 
        return masks
