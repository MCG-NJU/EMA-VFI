import os
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
from benchmark.utils.pytorch_msssim import ssim_matlab

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
down_scale = 0.5
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()


print(f'=========================Starting testing=========================')
print(f'Dataset: SNU_FILM   Model: {model.name}   TTA: {TTA}')
path = args.path
level_list = ['test-easy.txt', 'test-medium.txt', 'test-hard.txt', 'test-extreme.txt'] 
for test_file in level_list:
    psnr_list, ssim_list = [], []
    file_list = []
    
    with open(os.path.join(path, test_file), "r") as f:
        for line in f:
            line = line.strip()
            file_list.append(line.split(' '))

    for line in tqdm(file_list):
        I0_path = os.path.join(path, line[0])
        I1_path = os.path.join(path, line[1])
        I2_path = os.path.join(path, line[2])
        I0 = cv2.imread(I0_path)
        I1_ = cv2.imread(I1_path)
        I2 = cv2.imread(I2_path)
        I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
        I1 = (torch.tensor(I1_.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
        I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
        padder = InputPadder(I0.shape, divisor=32)
        I0, I2 = padder.pad(I0, I2)
        I1_pred = model.hr_inference(I0, I2, TTA, down_scale=down_scale, fast_TTA=TTA)[0]
        I1_pred = padder.unpad(I1_pred)
        ssim = ssim_matlab(I1, I1_pred.unsqueeze(0)).detach().cpu().numpy()

        I1_pred = I1_pred.detach().cpu().numpy().transpose(1, 2, 0)   
        I1_ = I1_ / 255.
        psnr = -10 * math.log10(((I1_ - I1_pred) * (I1_ - I1_pred)).mean())
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    
    print('Testing level:' + test_file[:-4])
    print('Avg PSNR: {} SSIM: {}'.format(np.mean(psnr_list), np.mean(ssim_list)))
