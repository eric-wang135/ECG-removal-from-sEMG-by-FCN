import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
# import sru

device = torch.device('cuda:0')


class Dense_L(nn.Module):

    def __init__(self, in_size, out_size,bias=True):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(in_size, out_size, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.dense(x)
        return out

class Conv(nn.Module):

    def __init__(self, in_chan, out_chan, kernal ,kernal_m=3, stride=1, dilation=1, padding=0, groups=1,dropout=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernal, stride=stride, dilation=dilation, padding=padding),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class conv_1d(nn.Module):
    def __init__(self,in_channel,out_channel,frame_size,shift):
        super().__init__()
        self.conv_1d = nn.Sequential(
        nn.Conv1d(in_channel,out_channel,frame_size,shift),
        nn.BatchNorm1d(out_channel),  
        nn.ELU(),
        )
    def forward(self, x):
        out = self.conv_1d(x)
        return out

class deconv_1d(nn.Module):
    def __init__(self,in_channel,out_channel,frame_size,shift,out_pad=0):
        super().__init__()
        self.deconv_1d = nn.Sequential(
        nn.ConvTranspose1d(in_channel,out_channel,frame_size,shift,output_padding=out_pad),
        nn.BatchNorm1d(out_channel),  
        nn.ELU(),
        )
    def forward(self, x):
        out = self.deconv_1d(x)
        return out


class FCN_01(nn.Module):
    #Larger latent space, less layer, wider
    def __init__(self,):
        super().__init__()
        self.frame_size = 16
        self.encoder = nn.Sequential(
            conv_1d(1,80,self.frame_size,2),
            conv_1d(80,40,self.frame_size,2),
            conv_1d(40,20,self.frame_size,1),
        )
        self.decoder = nn.Sequential(
            deconv_1d(20,20,self.frame_size,1),
            deconv_1d(20,40,self.frame_size,2),
            deconv_1d(40,80,self.frame_size,2),
            nn.ConvTranspose1d(80,1,self.frame_size,1),
        )
    def forward(self,emg):
        f = self.encoder(emg.unsqueeze(1))
        out = self.decoder(f)
        return out[:,:,:emg.shape[1]].squeeze(1)


