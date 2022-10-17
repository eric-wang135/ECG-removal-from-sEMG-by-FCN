import os, argparse, torch, random
from Trainer import Trainer
from Load_model import Load_model, Load_data
from util import check_folder
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pandas as pd
import pdb, sys

# fix random
SEED = 999
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True
# assign gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_path', type=str, default='./train_data_new2_f2')
    parser.add_argument('--test_noisy', type=str, default='./data_new2_1spk/test/noisy')
    parser.add_argument('--test_clean', type=str, default='./data_new2_f_band/test/clean')
    parser.add_argument('--writer', type=str, default='./train_log4')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)  
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--loss_fn', type=str, default='l1')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--model', type=str, default='DDAE_02') 
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--resume' , action='store_true', default=False)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--task', type=str, default='denoise')  # denoise / synthesis
    parser.add_argument('--val', action='store_true', default=True)
    parser.add_argument('--STFT', action='store_true', default=False)
    parser.add_argument('--inputdim_fix', action='store_true', default=False)
    parser.add_argument('--train_noisy', type=str, default='noisy')
    parser.add_argument('--train_clean', type=str, default='clean')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    print(SEED)
    print("Conda available:",torch.cuda.is_available())
    # get parameter
    args = get_args()

    # declair path
    
    checkpoint_path = f'./checkpoint/{args.task}_{args.model}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.pth.tar'
    model_path = f'./save_model/{args.task}_{args.model}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.pth.tar'
    
    score_path = f'./Result/{args.task}_{args.model}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.csv'
    
    # tensorboard
    writer = SummaryWriter(args.writer)
    # import model from its directory and create a model
    exec (f"from model.{args.model.split('_')[0]} import {args.model} as model")
    model     = model()

    model, epoch, best_loss, optimizer, criterion, device = Load_model(args,model,checkpoint_path,model)
    

    loader = Load_data(args) if args.mode == 'train' else 0
    print("Establish trainer")
    Trainer = Trainer(model, args.epochs, epoch, best_loss, optimizer, 
                      criterion, device, loader, writer, model_path, score_path,args)
    try:
        if args.mode == 'train':
            print("Training start")
            Trainer.train()
        Trainer.test()
        
 
    except KeyboardInterrupt:
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
            }

        check_folder(checkpoint_path)
        torch.save(state_dict, checkpoint_path)
        print('epoch:',epoch)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
