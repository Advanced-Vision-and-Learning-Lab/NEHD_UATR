# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:20:07 2021
Implementation of histogram layer with only convolutional layer and global average pooling
@author: jpeeples
"""

import torch
import torch.nn as nn
from Models.EHD import EHD
from Models.RBFHistogramPooling import HistogramLayer
import numpy as np
import os
import matplotlib.pyplot as plt

class NEHDLayer(nn.Module):
    def __init__(self,in_channels,window_size=[3,3],aggregation_type = 'GAP',
                 mask_size=3,num_bins=4, stride=1,padding=0,normalize_count=True,normalize_bins = True,
                 count_include_pad=False,ceil_mode=False,EHD_init=True,
                 learn_no_edge=True,learn_kernel= True,learn_hist=True,threshold=.9,angle_res=45,
                 normalize_kernel=False,
                 dilation=1, is_ehd = False, threshold_func= nn.Sigmoid()):

        # inherit nn.module
        super(NEHDLayer, self).__init__()

        # define layer properties
        # histogram bin data
        self.in_channels = in_channels
        self.num_bins = num_bins
        self.stride = stride
        self.window_size = window_size
        self.mask_size = mask_size
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode
        self.learn_no_edge= learn_no_edge
        self.threshold = threshold
        self.angle_res = angle_res
        self.normalize_kernel = normalize_kernel
        self.dilation = dilation
        self.aggregation_type = aggregation_type
        self.EHD_init = EHD_init
        self.learn_kernel = learn_kernel
        self.threshold_func = threshold_func
        self.learn_hist = learn_hist
        self.is_ehd = is_ehd
        self.feature_maps = {}
        #For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
        
        #If no edge transform should be learned
        if self.learn_no_edge:
            
            self.no_edge_conv = nn.Sequential(nn.Conv2d(self.in_channels*self.num_bins,self.in_channels,
                                              self.mask_size,groups=self.in_channels,
                                              bias=False,dilation=self.dilation,padding='same'),self.threshold_func)
            
        else:
            
            self.no_edge_conv = nn.Sequential(nn.Conv2d(self.in_channels*self.num_bins,self.in_channels,
                                              self.mask_size,groups=self.in_channels,
                                              bias=False,dilation=self.dilation),self.threshold_func)
            self.no_edge_conv[0].weight.requires_grad = False
        
        if self.in_channels == 1: #Gray scale (independently)
            self.edge_responses = nn.Conv2d(self.in_channels,self.num_bins,
                                              self.mask_size,groups=self.in_channels,
                                              bias=False,dilation=self.dilation)
            

        else: #Multichannel, aggregate
            self.edge_responses = nn.Conv2d(self.in_channels,self.num_bins*self.in_channels,
                                              self.mask_size,groups=self.in_channels,
                                              bias=False,dilation=self.dilation)
                
        bin_number = self.edge_responses.out_channels + self.in_channels    
            
        #Intialize histogram layer for aggregation of edge responses
        self.histogram_layer = HistogramLayer(self.edge_responses.out_channels, self.mask_size, dim=2,
                                              num_bins= (self.num_bins + 1), stride=self.stride,
                                              padding= self.padding, normalize_count= self.normalize_count,
                                              normalize_bins= self.normalize_bins,
                                              count_include_pad= self.count_include_pad,
                                              ceil_mode= self.ceil_mode)
        
        #Change histogram layer convolution to apply bin width and center to each individual feature map
        self.histogram_layer.bin_centers_conv = nn.Conv2d(bin_number,bin_number,1,
                                        groups=bin_number,bias=True)
        self.histogram_layer.bin_centers_conv.weight.data.fill_(1)
        self.histogram_layer.bin_centers_conv.weight.requires_grad = False
        self.histogram_layer.bin_widths_conv = nn.Conv2d(bin_number,
                                         bin_number,1, groups=bin_number,
                                         bias=False)
        
        #Change centers and widths of histogram to be shared with NEHD
        self.histogram_layer.centers =  self.histogram_layer.bin_centers_conv.bias
        self.histogram_layer.widths = self.histogram_layer.bin_widths_conv.weight 
        
        #Change aggregation type for histogram layer
        self.histogram_layer = self.set_histogram_layer()
          
        #Initialize edge kernels to correct values and intial bins to 0 and widths to 1
        if self.EHD_init:
            masks = EHD.Generate_masks(mask_size=self.mask_size,angle_res=self.angle_res,
                                   normalize= self.normalize_kernel,rotate=False)
            masks = torch.tensor(masks).float().unsqueeze(1)
            
            #Repeat along channel dimension
            masks = masks.repeat(1,self.in_channels,1,1)
            
            #Hard code for MNIST, change later (maximum value of input image)
            self.bin_init = 1
            self.bin_init = sum(2*self.bin_init*torch.topk(torch.flatten(masks[0,0]),self.mask_size[0])[0])
            masks.requires_grad = True
        
            if self.in_channels == 1: #Gray
                self.edge_responses.weight.data[0:self.num_bins] = masks
            else:
                
                #Generate indices for multichannel (treat each channel independently)
                indices = np.arange(0,self.in_channels*(self.num_bins))
                
                #Split into equal parts to match corresponding kernels
                indices = np.array_split(indices,self.in_channels)
                
                for channel in range(0,self.in_channels):
                    self.edge_responses.weight.data[indices[channel]]= masks[:,channel].unsqueeze(1)
              
          
            self.histogram_layer.bin_centers_conv.bias.data.fill_(-self.bin_init.clone().detach().requires_grad_(True))
            self.histogram_layer.bin_centers_conv.weight.data.fill_(1)
            self.histogram_layer.bin_centers_conv.weight.requires_grad = False
            
            #Started with 1 (for experiments), 3.75 for figure in paper
            self.histogram_layer.bin_widths_conv.weight.data.fill_(1)
     
        # Set the learning parameters for the kernel
        if self.learn_kernel:
            self.edge_responses.weight.requires_grad = True
        else:
            self.edge_responses.weight.requires_grad = False
         
        # Set the learning parameters for the histogram layer
        if self.learn_hist:
            self.histogram_layer.centers.requires_grad = True
            self.histogram_layer.widths.requires_grad = True
        else:
            self.histogram_layer.centers.requires_grad = False
            self.histogram_layer.widths.requires_grad = False
            
        #Set values for analysis later
        self.centers = self.histogram_layer.bin_centers_conv.bias
        self.widths = self.histogram_layer.bin_widths_conv.weight
        
        #If learning no edge, compute average mask for no edge orientation and 
        #concatenate it for visualization later
        if self.learn_no_edge:
            avg_threshold_mask = torch.mean(self.no_edge_conv[0].weight,dim=1,keepdim=True)
            self.edge_kernels = torch.cat((self.edge_responses.weight,avg_threshold_mask),dim=0)
        else:
            self.edge_kernels = self.edge_responses.weight
        
     
    def set_histogram_layer(self):
        #Perform global or local average pooling for feature
        if self.aggregation_type == 'GAP':
            self.histogram_layer.hist_pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.histogram_layer.hist_pool = nn.AvgPool2d(self.window_size,stride=self.stride,
                                             padding=self.padding,ceil_mode=self.ceil_mode,
                                             count_include_pad=self.count_include_pad)
        return self.histogram_layer
    
    def forward(self,xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window
        #Learn sobel responses
        self.feature_maps['feature_layer'] = xx
        xx = self.edge_responses(xx)
        self.feature_maps['edge_responses'] = xx
        #TO DO: remove if statement (call function or CNN layer for no edge)
        if not(self.learn_no_edge):
            xx_no_edge = self.get_no_edge(xx)
        else: #learn no edge through convolution
            xx_no_edge = self.no_edge_conv(xx)
       

        xx = torch.cat((xx,xx_no_edge),dim=1)
        self.feature_maps['no_edge'] = xx
        #Pass through histogram layer (binning)
        if self.is_ehd:
            # print("EHD")
            return xx
        xx_hist = self.histogram_layer(xx)
        # print("NEHD")
        self.feature_maps['histogram_layer'] = xx_hist

        return xx_hist
    
    #Will need to change for multi-channel, similar to constrain bins
    def get_no_edge(self,xx):
        # This function has been modified up due to an issue
        # When there is one function it does not work since
        # xx.shape = torch.Size([1, 8, 110, 110])
        # self.get_no_edge(xx).shape = torch.Size([1, 1, 1, 110, 110])
        # Therefore:
        # xx = torch.cat((xx,xx_no_edge),dim=1) produces an error saying function mismatch
        # One change was the [value, index] definition


        #Reshape tensor to be N x C x B x M x N 
        n,c,h,w = xx.size()
        xx_max = xx.reshape(n, c//self.num_bins, self.num_bins, h, w)
        
        #Find max response for each channel dimension
        if self.in_channels == 1:
            [value,index] = torch.max(xx,dim=1)
        else:
            [value,index] = torch.max(xx_max,dim=2)
            
        #Set edge responses to "no edge" if not larger than threshold
        if self.in_channels == 1:
            xx_no_edge = (value<self.threshold).float().unsqueeze(1)
        else:
            xx_no_edge = (value<self.threshold).float()
       
        return xx_no_edge
    
        
    def save_feature_maps(self, directory='./Feature_Maps/'):
        for layer_name, feature_map in self.feature_maps.items():
            os.makedirs(directory, exist_ok=True)
            layer_viz = torch.squeeze(feature_map, 0)
            print("Layer ", layer_name)
            print(layer_viz.shape)
            
            # Calculate global min and max values for the layer
            global_min = layer_viz.min().item()
            global_max = layer_viz.max().item()
            
            for i, f in enumerate(layer_viz):
                plt.imshow(f.detach().cpu().numpy(), cmap='viridis', vmin=global_min, vmax=global_max)
                plt.colorbar()
                plt.axis("off")
                plt.savefig(os.path.join(directory, f'{layer_name}_feature_map_{i}.png'), bbox_inches='tight', pad_inches=0)
                plt.close()
