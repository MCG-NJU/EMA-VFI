import os
import sys
import cv2
import math
import glob
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
from benchmark.utils.pytorch_msssim import ssim_matlab

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_t', type=str)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()
assert args.model in ['ours_t', 'ours_small_t'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
down_scale = 1.0
if args.model == 'ours_small_t':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )

model = Model(-1)
model.load_model()
model.eval()
model.device()

def getXVFI(dir, multiple=8, t_step_size=32):
    """ make [I0,I1,It,t,scene_folder] """
    testPath = []
    t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
    for type_folder in sorted(glob.glob(os.path.join(dir, '*', ''))):
        for scene_folder in sorted(glob.glob(os.path.join(type_folder, '*', ''))):
            frame_folder = sorted(glob.glob(scene_folder + '*.png'))
            for idx in range(0, len(frame_folder), t_step_size):
                if idx == len(frame_folder) - 1:
                    break
                for mul in range(multiple - 1):
                    I0I1It_paths = []
                    I0I1It_paths.append(frame_folder[idx])
                    I0I1It_paths.append(frame_folder[idx + t_step_size])
                    I0I1It_paths.append(frame_folder[idx + int((t_step_size // multiple) * (mul + 1))])
                    I0I1It_paths.append(t[mul])
                    testPath.append(I0I1It_paths)

    return testPath


print(f'=========================Starting testing=========================')
print(f'Dataset: X4K1000FPS   Model: {model.name}   TTA: {TTA}')
data_path = args.path
listFiles = getXVFI(data_path)
for strMode in ['XTEST-2k', 'XTEST-4k']:
    fltPsnr, fltSsim = [], []
    for intFrame in tqdm(listFiles):
        npyOne = np.array(cv2.imread(intFrame[0])).astype(np.float32) * (1.0 / 255.0)
        npyTwo = np.array(cv2.imread(intFrame[1])).astype(np.float32) * (1.0 / 255.0)
        npyTruth = np.array(cv2.imread(intFrame[2])).astype(np.float32) * (1.0 / 255.0)

        if strMode == 'XTEST-2k': #downsample
            down_scale = 0.5
            npyOne = cv2.resize(src=npyOne, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
            npyTwo = cv2.resize(src=npyTwo, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
            npyTruth = cv2.resize(src=npyTruth, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
        else:
            down_scale = 0.25

        tenOne = torch.FloatTensor(np.ascontiguousarray(npyOne.transpose(2, 0, 1)[None, :, :, :])).cuda()
        tenTwo = torch.FloatTensor(np.ascontiguousarray(npyTwo.transpose(2, 0, 1)[None, :, :, :])).cuda()
        tenGT = torch.FloatTensor(np.ascontiguousarray(npyTruth.transpose(2, 0, 1)[None, :, :, :])).cuda()

        padder = InputPadder(tenOne.shape, 32)
        tenOne, tenTwo = padder.pad(tenOne, tenTwo)
        tenEstimate = model.hr_inference(tenOne, tenTwo, TTA = TTA, timestep=intFrame[3], down_scale = down_scale)[0]
        tenEstimate = padder.unpad(tenEstimate)

        npyEstimate = (tenEstimate.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).round().astype(np.uint8)
        tenEstimate = torch.FloatTensor(npyEstimate.transpose(2, 0, 1)[None, :, :, :]).cuda() / 255.0

        fltPsnr.append(-10 * math.log10(torch.mean((tenEstimate - tenGT) * (tenEstimate - tenGT)).cpu().data))
        fltSsim.append(ssim_matlab(tenEstimate,tenGT).detach().cpu().numpy())

    print(f'{strMode}  PSNR: {np.mean(fltPsnr)}  SSIM: {np.mean(fltSsim)}')