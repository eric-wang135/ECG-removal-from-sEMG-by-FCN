import os,pdb,sys,math,librosa
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from numpy.core.defchararray import find
from scipy import signal
import scipy.optimize as spo
from scipy.stats.stats import pearsonr 

def check_path(path):
    # Check if path directory exists. If not, create a file directory
    if not os.path.isdir(path): 
        os.makedirs(path)
        
def check_folder(path):
    # Check if the folder of path exists
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)


def get_filepaths(directory,ftype='.npy'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)

def creat_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#------------------------------------------------------------------------

def resample(x, fs, fs_2):
    # x needs to be an 1D numpy array
    return signal.resample(x,int(x.shape[0]/fs * fs_2))

#------------------------------------------------------------------------

def cal_score(clean,enhanced,dtype='numpy'):
    if dtype == 'numpy':
        noise = enhanced - clean
        noise_pw = np.dot(noise,noise)
        signal_pw = np.dot(clean,clean)
        SNR = 10*np.log10(signal_pw/noise_pw)
    else:
        noise = enhanced - clean
        noise_pw = torch.sum(noise*noise,1)
        signal_pw = torch.sum(clean*clean,1)
        SNR = torch.mean(10*torch.log10(signal_pw/noise_pw)).item()
    return round(SNR,3)

def cal_rmse(clean,enhanced,dtype='numpy'):
    if dtype == 'numpy':
        RMSE = np.sqrt(((enhanced - clean) ** 2).mean())
    else:
        RMSE = torch.sqrt(torch.mean(torch.square(enhanced - clean))).item()
    return round(RMSE,6)
def cal_prd(clean,enhanced,dtype='numpy'):
    if dtype == 'numpy':
         PRD = np.sqrt(np.sum((enhanced - clean) ** 2) / np.sum(clean ** 2)) * 100
    else:
        PRD = torch.mul(torch.sqrt(torch.div(torch.sum(torch.square(enhanced - clean)),torch.sum(torch.square(clean)))),100).item()
    return round(PRD,3)

def cal_R2(clean,enhanced):
    R2 = pearsonr(clean,enhanced)[0]**2
    return round(R2,3)

def cal_CC(clean,enhanced):
    CC = np.correlate(clean,enhanced)[0]
    return round(CC,3)

def cal_ARV(emg):
  win = 1000
  ARV = []
  emg = abs(emg)
  for i in range(0,emg.shape[0],win):
    ARV.append((emg[i:i+win]).mean())
  return np.array(ARV)

def cal_KR(x):
  bins = np.linspace(-5,5,1001)
  pdf, _ = np.histogram(normalize(x),bins,density=True) # _ is bin
  cdf= np.cumsum(pdf)/np.sum(pdf)
  KR = (find_nearest(cdf,0.975)-find_nearest(cdf,0.025))/(find_nearest(cdf,0.75)-find_nearest(cdf,0.25))-2.91
  bin_centers = 0.5*(bins[1:] + bins[:-1])
  return KR

def cal_MF(emg,stimulus):
  # 10 - 500Hz mean frequency
  freq = librosa.fft_frequencies(sr=1000,n_fft=256)
  start = next(i for i,v in enumerate(freq) if v >=10)
  freq = np.expand_dims(freq[start:],1)
  spec = make_spectrum(emg,feature_type=None)[0][start:,:]
  weighted_f = np.sum(freq*spec,0)
  spec_column_pow = np.sum(spec,0)
  MF = weighted_f / spec_column_pow
  MF = [MF[i] for i,v in enumerate(stimulus[::32]) if v>0]
  return np.array(MF)


def normalize(x):
  return (x-x.mean())/np.std(x)

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

# ------------------------------------------------------------------------

def make_spectrum(y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=None,
                 SHIFT=None, _max=None, _min=None):
    
    D = librosa.stft(y,center=True, n_fft=256, hop_length=32, win_length=128,window=signal.hamming)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    ### Feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D
    ### normalizaiton mode
    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std  
    elif mode == 'minmax':
        _min = np.max(Sxx)
        _max = np.min(Sxx)
        Sxx = (Sxx - _min)/(_max - _min)

    return Sxx, phase, len(y)

def recons_spec_phase(Sxx_r, phase, length_wav, feature_type='logmag'):
    if feature_type == 'logmag':
        Sxx_r = np.expm1(Sxx_r)
        if np.min(Sxx_r) < 0:
            print("Expm1 < 0 !!")
        Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)
    elif feature_type == 'lps':
        Sxx_r = np.sqrt(10**Sxx_r)

    R = np.multiply(Sxx_r , phase)
    result = librosa.istft(R,
                     #center=True,
                     center=True,
                     hop_length= 32,
                     win_length= 128,
                     n_fft = 256,
                     window=signal.hamming,
                     length=length_wav,
                     )
    return result


# ------------------------------------------------------------------------
def progress_bar(epoch, epochs, step, n_step, time, loss, mode):
    line = []
    line = f'\rEpoch {epoch}/ {epochs}'
    loss = loss/step
    if step==n_step:
        progress = '='*30
    else :
        n = int(30*step/n_step)
        progress = '='*n + '>' + '.'*(29-n)
    eta = time*(n_step-step)/step
    line += f'[{progress}] - {step}/{n_step} |Time :{int(time)}s |ETA :{int(eta)}s  '
    if step==n_step:
        line += '\n'
    sys.stdout.write(line)
    sys.stdout.flush()

