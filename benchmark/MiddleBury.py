import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from utils.padder import InputPadder
from benchmark.utils.pytorch_msssim import ssim_matlab

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
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
print(f'Dataset: MiddleBury   Model: {model.name}   TTA: {TTA}')
path = args.path
name = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
IE_list = []
for i in name:
    i0 = cv2.imread(path + '/other-data/{}/frame10.png'.format(i)).transpose(2, 0, 1) / 255.
    i1 = cv2.imread(path + '/other-data/{}/frame11.png'.format(i)).transpose(2, 0, 1) / 255.
    gt = cv2.imread(path + '/other-gt-interp/{}/frame10i11.png'.format(i)) 
    i0 = torch.from_numpy(i0).unsqueeze(0).float().cuda()
    i1 = torch.from_numpy(i1).unsqueeze(0).float().cuda()
    padder = InputPadder(i0.shape, divisor = 32)
    i0, i1 = padder.pad(i0, i1)
    pred1 = model.inference(i0, i1, TTA=TTA, fast_TTA=TTA)[0]
    pred = padder.unpad(pred1)
    out = pred.detach().cpu().numpy().transpose(1, 2, 0)
    out = np.round(out * 255.)
    IE_list.append(np.abs((out - gt * 1.0)).mean())
print(f"Avg IE: {np.mean(IE_list)}")