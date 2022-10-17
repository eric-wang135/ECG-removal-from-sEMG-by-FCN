import torch.nn as nn
import torch.nn.functional as F
import torch
#import mkl
import os, sys, time, numpy as np, pandas as pd,pdb
from tqdm import tqdm
from scipy import signal
from util import *
from TS import *


class Trainer:
    def __init__(self, model, epochs, epoch, best_loss, optimizer, 
                      criterion, device, loader, writer, model_path, score_path, args):
#         self.step = 0
        self.epoch = epoch
        self.epoch_count = 0
        self.epochs = epochs
        self.best_loss = best_loss
        self.best_loss_snr = -100
        self.model = model.to(device)
        self.optimizer = optimizer

        self.device = device
        self.loader = loader
        self.criterion = criterion

        self.task = args.task
        self.train_loss = 0
        self.train_snr = 0
        self.val_loss = 0
        self.val_snr = 0
        self.writer = writer
        self.model_path = model_path
        self.score_path = score_path
        self.train_clean = args.test_clean.replace('test','train')
        self.inputdim_fix = args.inputdim_fix
        self.STFT = args.STFT
        self.out_folder = self.model.__class__.__name__
        if args.task == 'evaluate_HP':
            self.fc = 40
            self.highpass = signal.butter(4,self.fc,'highpass',fs=1000)

        if args.mode=='train':
            self.train_step = len(loader['train'])
            self.val_step = len(loader['val'])
        self.args = args
           
    def save_checkpoint(self,):
        state_dict = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
            }
        check_folder(self.model_path)
        torch.save(state_dict, self.model_path)

    def print_score(self,test_file):
        self.model.eval()
        n_emg = torch.load(test_file)
        c_file = os.path.join(self.train_clean,test_file.split('/')[-1].replace('.pt','.npy'))
        c_emg = np.load(c_file)
        pred = self.model(n_emg)
        loss = self.criterion(pred, c_emg).item()
        pred = pred.cpu().detach().numpy()

    def _train_epoch(self):
        self.train_loss = 0
        self.train_snr = 0
        self.model.train()
        t_start =  time.time()
        step = 0
        self._train_step = getattr(self,f'_train_step_mode_{self.task}')
        for data in self.loader['train']:
            step += 1
            self._train_step(data)
            progress_bar(self.epoch,self.epochs,step,self.train_step,time.time()-t_start,loss=self.train_loss,mode='train')
            
        self.train_loss /= len(self.loader['train'])
        self.train_snr /= len(self.loader['train'])
        print(f'train_loss:{self.train_loss}')
        print(f'train_snr:{self.train_snr}')
#     @torch.no_grad()
    
    def _val_epoch(self):
        self.val_loss = 0
        self.val_snr = 0
        self.model.eval()
        t_start =  time.time()
        step = 0
        self._val_step = getattr(self,f'_val_step_mode_{self.task}')
        for data in self.loader['val']:
            step += 1
            self._val_step(data)
            progress_bar(self.epoch,self.epochs,step,self.val_step,time.time()-t_start,loss=self.val_loss,mode='val')
        self.val_loss /= len(self.loader['val'])
        self.val_snr /= len(self.loader['val'])
        print(f'val_loss:{self.val_loss}')
        print(f'val_snr:{self.val_snr}')
        if self.best_loss > self.val_loss:
            self.epoch_count = 0
            print(f"Save model to '{self.model_path}'")
            
            self.save_checkpoint()
            self.best_loss = self.val_loss
            self.best_loss_snr = self.val_snr

    def train(self):
        model_name = self.model.__class__.__name__ 
        while self.epoch < self.epochs and self.epoch_count<15:
            self._train_epoch()
            self._val_epoch()
            self.writer.add_scalars(f'{self.args.task}/{model_name}_{self.args.optim}_{self.args.loss_fn}', {'train': self.train_loss},self.epoch)
            self.writer.add_scalars(f'{self.args.task}/{model_name}_{self.args.optim}_{self.args.loss_fn}', {'val': self.val_loss},self.epoch)
            self.epoch += 1
            self.epoch_count += 1
        print("best loss:",self.best_loss)
        print("best loss snr:",self.best_loss_snr)
        self.writer.close()
    
    
    def write_score(self,test_file,test_path,output=False):   
        self.model.eval()
        outname  = test_file.replace(f'{test_path}','').replace('/','_')
        c_file = os.path.join(self.args.test_clean,test_file.split('/')[-1])
        clean = np.load(c_file)
        stimulus = np.load(c_file.replace('.npy','_sti.npy'))
        noisy = np.load(test_file)
        if self.args.task=='denoise':
            if self.args.STFT:
                n_emg,n_phase,n_len = make_spectrum(y = noisy)
                n_emg = torch.from_numpy(noisy).t()
                c_emg = make_spectrum(y = clean)[0]
                n_emg = torch.from_numpy(noisy).to(self.device).unsqueeze(0).type(torch.float32)
                c_emg = torch.from_numpy(c_emg).t().to(self.device).unsqueeze(0).type(torch.float32)
            else: 
                n_emg = torch.from_numpy(noisy).to(self.device).unsqueeze(0).type(torch.float32)
                c_emg = torch.from_numpy(clean).to(self.device).unsqueeze(0).type(torch.float32)
            
            if self.inputdim_fix==True:
                n_emg = F.pad(n_emg,(0,2000-n_emg.shape[1]%2000)).view(-1,2000)
                c_emg = F.pad(c_emg,(0,2000-c_emg.shape[1]%2000))
            
            pred = self.model(n_emg)

            if self.inputdim_fix==True:
                pred = pred.view(-1,1).squeeze()
                c_emg = c_emg.squeeze()
            loss = self.criterion(pred,c_emg).item()
            pred = pred.cpu().detach().numpy()
            enhanced = pred.squeeze()
            
        elif self.args.task=='evaluate':
            enhanced = noisy
            output = False
            error = 0

        elif self.args.task=='evaluate_HP':
            enhanced = signal.filtfilt(self.highpass[0],self.highpass[1],noisy).astype('float64')
            output = True
            error = 0
            self.out_folder = 'HP'+str(self.fc)
                    
        elif self.args.task=='evaluate_ATS':
            enhanced,error = template_subtraction(noisy)
            enhanced = enhanced.astype('float64')
            output = True 
            self.out_folder = 'ATS'

        elif self.args.task=='evaluate_FTS':
            enhanced,error = filtered_template_subtraction(noisy,50)
            enhanced = enhanced.astype('float64')
            output = True
            self.out_folder = 'FTS'

        elif self.args.task=='evaluate_FTSHP':
            highpass = signal.butter(4,40,'highpass',fs=1000)
            enhanced,error = filtered_template_subtraction(noisy,50)#template_subtraction(noisy)
            enhanced = signal.filtfilt(highpass[0],highpass[1],enhanced)
            enhanced = enhanced.astype('float64')
            output = True
            self.out_folder = 'HP20TS'

        # Evaluation metrics
        SNR = cal_score(clean,enhanced)
        RMSE = cal_rmse(clean,enhanced)
        PRD = cal_prd(clean,enhanced)
        RMSE_ARV = cal_rmse(cal_ARV(clean),cal_ARV(enhanced))
        KR = abs(cal_KR(clean)-cal_KR(enhanced))
        MF = cal_rmse(cal_MF(clean,stimulus),cal_MF(enhanced,stimulus))
        R2 = cal_R2(clean,enhanced)
        CC = cal_CC(clean,enhanced)
        if self.args.task == 'denoise':
            with open(self.score_path, 'a') as f1:
                f1.write(f'{outname},{SNR},{loss},{RMSE},{PRD},{RMSE_ARV},{KR},{MF},{R2},{CC}\n')
        else:
            with open(self.score_path, 'a') as f1:
                f1.write(f'{outname},{SNR},{error},{RMSE},{PRD},{RMSE_ARV},{KR},{MF},{R2},{CC}\n')
        if output:
            emg_path = test_file.replace(f'{test_path}',f'./enhanced_data_E2_S40_Ch11_nsrd/{self.out_folder}') 
            check_folder(emg_path)
            np.save(emg_path,enhanced)
       
    
            
    def test(self):
        # load model
        self.model.eval()
        print("best loss:",self.best_loss)
        if self.args.task == 'denoise':
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model'])
            test_path = self.args.test_noisy if self.args.task=='denoise' else self.args.test_clean
        else:
            test_path = self.args.test_noisy
        test_folders = get_filepaths(test_path)
        
        check_folder(self.score_path)
        if os.path.exists(self.score_path):
            os.remove(self.score_path)
        with open(self.score_path, 'a') as f1:
            f1.write('Filename,SNR,Loss,RMSE,PRD,RMSE_ARV,KR,MF,R2,CC\n')
        for test_file in tqdm(test_folders):
            self.write_score(test_file,test_path,output=True)
        
        data = pd.read_csv(self.score_path)
        snr_mean = data['SNR'].to_numpy().astype('float').mean()
        loss_mean = data['Loss'].to_numpy().astype('float').mean()
        rmse_mean = data['RMSE'].to_numpy().astype('float').mean()
        prd_mean = data['PRD'].to_numpy().astype('float').mean()
        arv_mean = data['RMSE_ARV'].to_numpy().astype('float').mean()
        kr_mean = data['KR'].to_numpy().astype('float').mean()
        mf_mean = data['MF'].to_numpy().astype('float').mean()
        r2_mean = data['R2'].to_numpy().astype('float').mean()
        cc_mean = data['CC'].to_numpy().astype('float').mean()
        with open(self.score_path, 'a') as f:
            f.write(','.join(('Average',str(snr_mean),str(loss_mean),str(rmse_mean),str(prd_mean),str(arv_mean),str(kr_mean),str(mf_mean),str(r2_mean),str(cc_mean)))+'\n')
        
    def _train_step_mode_denoise(self, data):
        device = self.device
        noisy, clean = data
        noisy, clean = noisy.to(device).type(torch.float32), clean.to(device).type(torch.float32)
        pred = self.model(noisy)
        loss = self.criterion(pred, clean)
        
        snr = cal_score(pred, clean, torch)
        self.train_loss += loss.item()
        self.train_snr += snr
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()    
        
    def _val_step_mode_denoise(self, data):
        device = self.device
        noisy, clean = data
        noisy, clean = noisy.to(device).type(torch.float32), clean.to(device).type(torch.float32)
        pred = self.model(noisy)
        loss = self.criterion(pred, clean)
        snr = cal_score(pred, clean, torch)
        self.val_snr += snr
        self.val_loss += loss.item()

    


    
