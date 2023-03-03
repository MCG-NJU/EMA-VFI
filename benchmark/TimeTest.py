from time import time
import sys
import torch
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_small', type=str)
parser.add_argument('--H', default=256, type=int)
parser.add_argument('--W', default=256, type=int)
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

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

H, W = args.H, args.W
I0 = torch.rand(1, 3, H, W).cuda()
I1 = torch.rand(1, 3, H, W).cuda()

print(f'Test model: {model.name}  TTA: {TTA}')
with torch.no_grad():
    for i in range(50):
        pred = model.inference(I0, I1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_stamp = time()
    for i in range(100):
        pred = model.inference(I0, I1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print((time() - time_stamp) / 100 * 1000)