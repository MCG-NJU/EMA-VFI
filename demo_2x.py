import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
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


print(f'=========================Start Generating=========================')

I0 = cv2.imread('example/img1.jpg')
I2 = cv2.imread('example/img2.jpg') 

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)

mid = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
mimsave('example/out_2x.gif', images, fps=3)


print(f'=========================Done=========================')