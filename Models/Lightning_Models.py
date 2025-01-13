"""
Created on Thursday June 20 12:25:00 2024
Wrap models in a PyTorch Lightning Module for training and evaluation
@author: salimalkharsa, a_agashe
"""

import torch.nn as nn
import lightning.pytorch as L
import os
import torch
import numpy as np
from Models.EHD import EHD
from Utils.Generate_Plots import plot_histogram, plot_kernels
import timm
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
from Models.PANN_models import Cnn14
from transformers import ViTImageProcessor, ASTModel, ASTConfig


class NEHD_Lightning(L.LightningModule):
    def __init__(self, model, epochs, in_channels, out_channels, angle_res, normalize_kernel, mask_size, filename):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.epochs = epochs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.angle_res = angle_res
        self.filename = filename
        self.feature_masks = EHD.Generate_masks(mask_size=mask_size,
                                                      angle_res=angle_res,
                                                      normalize=normalize_kernel)
        self.saved_bins = np.zeros(
            (epochs+1, int(in_channels) * int(out_channels)))
        self.saved_widths = np.zeros(
            (epochs+1, int(in_channels) * int(out_channels)))
        self.saved_bins[0, :] = self.model.histogram_layer.centers.reshape(
            -1).detach().cpu().numpy()
        self.saved_widths[0, :] = self.model.histogram_layer.widths.reshape(
            -1).detach().cpu().numpy()
        plot_histogram(self.saved_bins[0, :], self.saved_widths[0,
                       :], -1, 'train', self.angle_res, self.filename)
        self.epoch_weights = []

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_end(self):
        self.saved_bins[self.current_epoch + 1,
                        :] = self.model.histogram_layer.centers.detach().cpu().numpy()
        self.saved_widths[self.current_epoch + 1,
                          :] = self.model.histogram_layer.widths.reshape(-1).detach().cpu().numpy()

        self.epoch_weights.append(self.model.edge_kernels.data)
        plot_kernels(self.feature_masks, self.model.edge_kernels.data,
                     'train', self.current_epoch, self.in_channels, self.angle_res, self.filename)
        plot_histogram(self.saved_bins[self.current_epoch + 1, :], self.saved_widths[self.current_epoch +
                       1, :], self.current_epoch, 'train', self.angle_res, self.filename)

class Histogram_Lightning(L.LightningModule):
    def __init__(self, model, epochs, num_bins, filename):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        # return HistogramLayer(in_channels=network_parameters['in_channels'],
        #         kernel_size= network_parameters['window_size'],
        #         num_bins=network_parameters['num_bins'],
        #         stride=network_parameters['stride'],
        #         normalize_count=True,
        #         normalize_bins=True)
        self.model = model
        self.num_bins = num_bins
        self.filename = filename
        self.epochs = epochs
        self.saved_bins = np.zeros((self.epochs + 1, self.num_bins))
        self.saved_widths =  np.zeros((self.epochs + 1, self.num_bins))
        self.saved_bins[0,:] = self.model.centers.reshape(-1).detach().cpu().numpy() 
        self.saved_widths[0,:] = self.model.widths.reshape(-1).detach().cpu().numpy()

        plot_histogram(self.saved_bins[0, :], self.saved_widths[0,
                       :], -1, 'train', self.num_bins, self.filename)

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_end(self):


        self.saved_bins[self.current_epoch + 1, :] = self.model.centers.detach().cpu().numpy()
        self.saved_widths[self.current_epoch + 1, :] = self.model.widths.reshape(-1).detach().cpu().numpy()
        plot_histogram(self.saved_bins[self.current_epoch + 1, :], self.saved_widths[self.current_epoch +
                       1, :], self.current_epoch, 'train', self.num_bins, self.filename)


class Custom_TIMM_Model(L.LightningModule):
    def __init__(self, num_input_features, num_classes, model_name='resnet50'):
        super(Custom_TIMM_Model,self).__init__()
        self.model_name = model_name
        self.weights_path = {'resnet50': {'class': 'resnet50', 'path': f'./pretrained_models/resnet50_{num_input_features}.pth'},
                        'convnext': {'class': 'convnextv2_tiny.fcmae', 'path': f'./pretrained_models/convnextv2_tiny_fcmae_{num_input_features}.pth'},
                        'vit' : {'class': 'vit_small_patch16_224', 'path' :f'./pretrained_models/vit_small_{num_input_features}.pth'}}
        
        self.num_input_features = num_input_features
        self.in_chans = self.num_input_features
        self. num_classes = num_classes
        self._get_model()

    def _get_model(self):
        model_class = self.weights_path[self.model_name]['class']
        path = self.weights_path[self.model_name]['path']
        print(model_class)
        if self.training:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                print(f"pretrained model DOES NOT EXIST at ${path}")
                folder_name = "pretrained_models"
                try:
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                except Exception as e:
                    print(f"An error occurred: {e}")
                self.model = timm.create_model(model_class, pretrained=True, in_chans = self.num_input_features, num_classes = self. num_classes)
                torch.save(self.model.state_dict(), path)
            else:
                print(f"pretrained model EXISTS exist at ${path}")
                self.model = timm.create_model(model_class, pretrained=False, in_chans = self.num_input_features, num_classes = self. num_classes)
                state_dict = torch.load(path)
                self.model.load_state_dict(state_dict, strict=True)
            
        else:
            self.model = timm.create_model(model_class, pretrained=False, in_chans = self.num_input_features, num_classes = self. num_classes)


    def forward(self, features):
        if self.model_name == 'vit':

            target_height = 224
            target_width = 224
            _, _, current_height, current_width = features.shape

            pad_height_total = max(target_height - current_height, 0)
            pad_width_total = max(target_width - current_width, 0)
            pad_top = pad_height_total // 2
            pad_bottom = pad_height_total - pad_top
            pad_left = pad_width_total // 2
            pad_right = pad_width_total - pad_left
            features = F.pad(features, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)  # You can change the mode to 'constant', 'reflect', 'replicate', etc.

            # features = F.interpolate(features, size=(target_height,target_width), mode='bilinear', align_corners=False)

        outputs = self.model(features)
        return outputs





class PANN_Lightning(L.LightningModule):
    def __init__(self, num_input_features, num_classes):
        super(PANN_Lightning,self).__init__()
        self.save_hyperparameters(ignore=['self.model'])
        self._init_model(num_input_features, num_classes)
        self.padding_layer = nn.ConstantPad2d(padding=(12, 12, 2, 2), value=0)
        self.num_input_features = num_input_features
    def forward(self, x):
        if self.num_input_features > 1:
            x = self.padding_layer(x)
        # print(x.shape)
        return self.model(x)

    def _init_model(self, num_input_features, num_classes):

        # PANN models
        model_class = Cnn14
        weights_url = "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"

        weights_name = "Cnn14_16k_mAP=0.438.pth"
        weights_path = f"./pretrained_models/{weights_name}"
        folder_name = "pretrained_models"
        try:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
        except Exception as e:
            print(f"An error occurred: {e}")
        self.model = model_class(input_fts=num_input_features, classes_num=4)
        if self.training:
            if not os.path.exists(weights_path) or os.path.getsize(weights_path) == 0:
                download_weights(weights_url, weights_path)
            try:
                pretrained_weights = torch.load(weights_path)
                self.model.load_state_dict(pretrained_weights, strict=False)
                if model_class == Cnn14:
                    with torch.no_grad():
                        pretrained_conv1_weights = pretrained_weights["model"]['conv_block1.conv1.weight']
                        num_new_channels = num_input_features
                        avg_weights = pretrained_conv1_weights.mean(
                            dim=1, keepdim=True)
                        new_conv1_weights = avg_weights.repeat(
                            1, num_new_channels, 1, 1)
                        self.model.conv_block1.conv1.weight.data.copy_(
                            new_conv1_weights)
                print("\nPretrained PANN\n")
            except Exception as e:
                raise RuntimeError(f"Error loading the model weights: {e}")
        num_ftrs = self.model.fc_audioset.in_features
        self.model.fc_audioset = nn.Linear(num_ftrs, num_classes)
        if not self.training:
            for param in self.model.parameters():
                param.requires_grad = False

def download_weights(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading weights from {url} to {destination}...\n")
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print("Download complete.\n")
    else:
        print(f"Weights already exist at {destination}.\n")


# class StatStructModel(nn.Module):
#     def __init__(self,model,num_ftrs,num_classes,is_intermediate=False):

#         super(StatStructModel,self).__init__()
#         self.num_ftrs = num_ftrs
#         self.num_classes = num_classes
#         self.is_intermediate = is_intermediate
#         #Define neural feature
#         self.model = model
        
#         if is_intermediate:
#             self.fc = torch.nn.Sequential()
#             self.padding=(17, 17, 2, 2)
#         else:
#             self.fc = nn.Linear(num_ftrs, num_classes)
        
        
#     def forward(self,x):
#         # Preprocess
#         # [128,1,28,28]
#         #Extract features from histogram layer and pass to fully connected layer
#         # print(x.shape)
#         if self.model:
#             x = self.model(x)
#         #if reconstructon experiments, do not flatten tensor
#         #else classification, flatten tensor
#         if self.is_intermediate:
#             output = F.pad(x, self.padding, mode='constant', value=0)
#         else:
#             x = torch.flatten(x,start_dim=1)
#             output = self.fc(x)
#         return output
    
class LinearNetwork(L.LightningModule):
    def __init__(self,num_ftrs,num_classes):

        super(LinearNetwork,self).__init__()
        self.num_ftrs = num_ftrs
        self.num_classes = num_classes

        self.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self,x):
        x = torch.flatten(x,start_dim=1)
        return self.fc(x)


class AST(L.LightningModule):
    def __init__(self, max_length: int, num_input_features:int, num_classes: int, final_output: str):
        super(AST,self).__init__()

        assert final_output in ['CLS','ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
        self.weights_path = './pretrained_models/astmodel.pth'
        self.config_path = './pretrained_models/astmodelconfig'
        self.num_input_features = num_input_features
        if num_input_features > 1:
            self.conv1x1 = nn.Conv2d(in_channels=num_input_features, out_channels=1, kernel_size=1)
        if not os.path.exists(self.weights_path) or os.path.getsize(self.weights_path) == 0:
            self.model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length = max_length, ignore_mismatched_sizes=True)
            torch.save(self.model.state_dict(), self.weights_path)
            self.model.config.save_pretrained(self.config_path)
        
        else:
            config = ASTConfig.from_pretrained('./pretrained_models/astmodelconfig')
            self.model = ASTModel(config)
            self.model.load_state_dict(torch.load(self.weights_path))

        self.final_output = final_output

        self.classification_head = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, x):
        if self.num_input_features > 1:
            x = self.conv1x1(x)
        # print("Before Conv:",outputs.shape)
        output_size = (212, 40)
        # print(features.shape)
        # Perform interpolation to resize the tensor
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        x = x.squeeze(1)
        hidden_states = self.model(x)[0]
        
        if self.final_output == 'CLS':
            return self.classification_head(hidden_states[:,0])
        else:
            return self.classification_head(hidden_states.mean(dim=1))

    def forward_tsne(self,x):
        hidden_states = self.model(x)[0]
        return hidden_states[:,0], hidden_states.mean(dim=1)