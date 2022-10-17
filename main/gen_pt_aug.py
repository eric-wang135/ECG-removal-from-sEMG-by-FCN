import argparse,os,torch,numpy as np
from tqdm import tqdm
from util import *
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_path', type=str, default='data_E1_S40_Ch2_withSTI_seg60s_nsrd/train/noisy')
    parser.add_argument('--clean_path', type=str,default='data_E1_S40_Ch2_withSTI_seg60s_nsrd/train/clean')
    parser.add_argument('--out_path', type=str, default='./trainpt_E1_S40_Ch2_withSTI_seg60s_nsrd/')
    parser.add_argument('--frame_size', type=int, default=2000)
    parser.add_argument('--augment', type=int, default=1)

    parser.add_argument('--only_clean', action='store_true', default=False)
    parser.add_argument('--val', action='store_true', default=True)
    parser.add_argument('--STFT', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train_path = args.noisy_path
    clean_path = args.clean_path
    out_path = args.out_path    
    n_frame = args.frame_size
    # Generate noisy training file
    noisy_files = get_filepaths(train_path)
    for emg_file in tqdm(noisy_files):
        if args.only_clean:
            break
        emg_name = emg_file.split('/')[-1]
        noise = emg_file.split(os.sep)[-2]
        snr = emg_file.split(os.sep)[-3]
        nout_path = os.path.join(out_path,'noisy',snr,noise,emg_name.split(".")[0])
        n_emg = np.load(emg_file)
        if args.STFT:
            n_emg = torch.from_numpy(make_spectrum(y=n_emg)[0]).t()
        else:
            n_emg = torch.from_numpy(n_emg)
        for c in range(args.augment):
            for i in np.arange(int(n_emg.shape[0]-c*n_frame//args.augment)//n_frame):
                nout_name = nout_path+'_'+str(i)+'_'+str(c)+'.pt'
                check_folder(nout_name)
                torch.save(n_emg[int(c*n_frame//args.augment+i*n_frame):int(c*n_frame//args.augment+(i+1)*n_frame)].clone() ,nout_name)
    if args.val:
        noisy_files = get_filepaths(train_path.replace('train','val'))
        for emg_file in tqdm(noisy_files):
            if args.only_clean:
                break
            emg_name = emg_file.split('/')[-1]
            noise = emg_file.split(os.sep)[-2]
            snr = emg_file.split(os.sep)[-3]
            nout_path = os.path.join(out_path,'val',snr,noise,emg_name.split(".")[0])
            n_emg = np.load(emg_file)
            if args.STFT:
                n_emg = torch.from_numpy(make_spectrum(y=n_emg)[0]).t()
            else:
                n_emg = torch.from_numpy(n_emg)
            for c in range(args.augment): # Data Augmentation
                for i in np.arange(int(n_emg.shape[0]-c*n_frame//args.augment)//n_frame):
                    nout_name = nout_path+'_'+str(i)+'_'+str(c)+'.pt'
                    check_folder(nout_name)
                    torch.save(n_emg[int(c*n_frame//args.augment+i*n_frame):int(c*n_frame//args.augment+(i+1)*n_frame)].clone(),nout_name)
    # Generate clean training set
    clean_files = get_filepaths(clean_path)
    # Iterate through all file names in filepath "clean_files"
    for emg_file in tqdm(clean_files):
        emg_name = emg_file.split('/')[-1]
        cout_path = os.path.join(out_path,'clean')
        c_emg = np.load(emg_file)
        if args.STFT:
            c_emg = torch.from_numpy(make_spectrum(y=c_emg)[0]).t()
        else:
            c_emg = torch.from_numpy(c_emg)
        for c in range(args.augment):
            for i in np.arange(int(c_emg.shape[0]-c*n_frame//args.augment)//n_frame):
                # Save each segment of data(emma+spec) with n_frame by name folder/emg_name_i.pt
                cout_name = os.path.join(cout_path,emg_name.split(".")[0]+'_'+str(i)+'_'+str(c)+'.pt')
                # Create a folder with cout_path if not exist
                check_folder(cout_name)
                # Save emg data by n_frame
                torch.save( c_emg[int(c*n_frame//args.augment+i*n_frame):int(c*n_frame//args.augment+(i+1)*n_frame)].clone() ,cout_name)
    if args.val:
        clean_files = get_filepaths(clean_path.replace('train','val'))
        for emg_file in tqdm(clean_files):
            emg_name = emg_file.split('/')[-1]
            cout_path = os.path.join(out_path,'clean')
            c_emg = np.load(emg_file)
            if args.STFT:
                c_emg = torch.from_numpy(make_spectrum(y=c_emg)[0]).t()
            else:
                c_emg = torch.from_numpy(c_emg)
            for c in range(args.augment):
                for i in np.arange(int(c_emg.shape[0]-c*n_frame//args.augment)//n_frame):
                    cout_name = os.path.join(cout_path,emg_name.split(".")[0]+'_'+str(i)+'_'+str(c)+'.pt')
                    check_folder(cout_name)
                    torch.save( c_emg[int(c*n_frame//args.augment+i*n_frame):int(c*n_frame//args.augment+(i+1)*n_frame)].clone() ,cout_name)
                    
