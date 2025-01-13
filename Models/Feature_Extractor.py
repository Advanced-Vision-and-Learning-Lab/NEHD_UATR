"""
Created on Fri Jul 12 10:19:34 2019
Create Spectrogram Extractor
@author: Sat Jan 10
"""

import torch.nn as nn
from nnAudio import features

class Feature_Extraction_Layer(nn.Module):
    def __init__(self, audio_feature, sample_rate=16000, window_length=6144, 
                 hop_length=4096, freq_bins = 192, padding = [0,0,0,0]):
        super(Feature_Extraction_Layer, self).__init__()

        
        self.audio_feature = audio_feature
        

        print(f"n_fft={window_length}")
        print(f"hop_length={hop_length}")
        print(f"win_length={window_length}")
        print(f"freq_bins={freq_bins}")
        print(f"sample_rate={sample_rate}")

        self.feature_extractor = self.get_feature_extractor(audio_feature, sample_rate, window_length, hop_length, freq_bins, padding)

        
    def forward(self, x):
       
        x = self.feature_extractor(x).unsqueeze(1)

        return x

    def get_feature_extractor(self, audio_feature, sample_rate, window_length, hop_length, freq_bins, padding):

        
        if audio_feature == 'STFT':
            inherent_padding = [0,0,0,0]
            final_padding = map(lambda x: sum(x), zip(inherent_padding, padding))
            return nn.Sequential(features.STFT(sr=sample_rate,n_fft=int(window_length), 
                                hop_length=int(hop_length),
                                win_length=int(window_length), 
                                output_format='Magnitude',
                                freq_bins=freq_bins,verbose=False), nn.ZeroPad2d(tuple(final_padding)))
