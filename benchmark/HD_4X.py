import os
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2yuv
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
from benchmark.utils.yuv_frame_io import YUV_Read

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_t', type=str)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()
assert args.model in ['ours_t', 'ours_small_t'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
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

print(f'=========================Starting testing=========================')
print(f'Dataset: HD_4X   Model: {model.name}   TTA: {TTA}')
path = args.path
name_list = [
    ('HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280),
    ('HD720p_GT/shields_1280x720_60.yuv', 720, 1280),
    ('HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280),
    ('HD1080p_GT/BlueSky.yuv', 1080, 1920),
    ('HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920),
    ('HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920),
    ('HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920),
    ('HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280),
    ('HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280),
    ('HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280),
    ('HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280),
]

        
tot = []
for data in tqdm(name_list):
    psnr_list = []
    name = data[0]
    h, w = data[1], data[2]
    Reader = YUV_Read(os.path.join(path, name), h, w, toRGB=True)
    _, lastframe = Reader.read()

    for index in range(0, 100, 4):
        gt = []
        IMAGE1, success1 = Reader.read(index)
        IMAGE2, success2 = Reader.read(index + 4)
        if not success2:
            break
        for i in range(1, 4):
            tmp, _ = Reader.read(index + i)
            gt.append(tmp)

        I0 = torch.from_numpy(np.transpose(IMAGE1, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
        I1 = torch.from_numpy(np.transpose(IMAGE2, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
        
        padder = InputPadder(I0.shape, divisor=32)
        I0, I1 = padder.pad(I0, I1)
        pred_list = model.multi_inference(I0, I1, TTA=TTA, time_list=[(i+1)*(1./4.) for i in range(3)], fast_TTA = TTA)
        for i in range(len(pred_list)):
            pred_list[i] = padder.unpad(pred_list[i])

        for i in range(3):
            out = (np.round(pred_list[i].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')
            diff_rgb = 128.0 + rgb2yuv(gt[i] / 255.)[:, :, 0] * 255 - rgb2yuv(out / 255.)[:, :, 0] * 255
            mse = np.mean((diff_rgb - 128.0) ** 2)
            PIXEL_MAX = 255.0
            psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

            psnr_list.append(psnr)

    tot.append(np.mean(psnr_list))

print('PSNR: {}(544*1280), {}(720p), {}(1080p)'.format(np.mean(tot[7:11]), np.mean(tot[:3]), np.mean(tot[3:7])))
