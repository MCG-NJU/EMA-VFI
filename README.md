# Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation [arxiv](https://arxiv.org/abs/2303.00440)

> [**Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation**](https://arxiv.org/abs/2303.00440)<br>
> Accepted by **CVPR 2023**<br>
> [Guozhen Zhang](https://github.com/GuozhenZhang1999), [Yuhan Zhu](https://github.com/Breeze-zyuhan), [Haonan Wang](https://github.com/haonanwang0522), Youxin Chen, [Gangshan Wu](http://mcg.nju.edu.cn/member/gswu/en/index.html), [Limin Wang](http://wanglimin.github.io/)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-vimeo90k)](https://paperswithcode.com/sota/video-frame-interpolation-on-vimeo90k?p=extracting-motion-and-appearance-via-inter)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-ucf101-1)](https://paperswithcode.com/sota/video-frame-interpolation-on-ucf101-1?p=extracting-motion-and-appearance-via-inter)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-xiph-4k-1)](https://paperswithcode.com/sota/video-frame-interpolation-on-xiph-4k-1?p=extracting-motion-and-appearance-via-inter)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-xiph-2k)](https://paperswithcode.com/sota/video-frame-interpolation-on-xiph-2k?p=extracting-motion-and-appearance-via-inter)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-x4k1000fps-2k)](https://paperswithcode.com/sota/video-frame-interpolation-on-x4k1000fps-2k?p=extracting-motion-and-appearance-via-inter)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-x4k1000fps)](https://paperswithcode.com/sota/video-frame-interpolation-on-x4k1000fps?p=extracting-motion-and-appearance-via-inter)

<div align="center">
  <img src="figs/EMA-VFI.png" width="1000"/>
</div>

## :boom: News

* **[2023.03.12] We compared our method with other methods (VFIFormer and M2M) under extreme cases such as large motion and scene transitions. The video demonstrating our results can be found [here](https://www.youtube.com/watch?v=ziJLIPzu2HI) ([Bilibili](https://www.bilibili.com/video/BV1584y1A7ae/?vd_source=92c28105d36256b9f950b8793336df88)).**
* **[2023.03.12] Thanks to @[jhogsett](https://github.com/jhogsett), our model now has a more user-friendly [WebUI](https://github.com/jhogsett/EMA-VFI-WebUI)!**

## :satisfied: HighLights

In this work, we propose to exploit inter-frame attention for extracting motion and appearance information in video frame interpolation. In particular, we utilize the correlation information hidden within the attention map to simultaneously enhance the appearance information and model motion. Meanwhile, we devise an hybrid CNN and Transformer framework to achieve a better trade-off between performance and efficiency. Experiment results show that our proposed module achieves state-of-the-art performance on both fixed- and arbitrary-timestep interpolation and enjoys effectiveness compared with the previous SOTA method.

Runtime and memory usage compared with previous SOTA method:
<div align="center">
  <img src=figs/time.png width=400 />
</div>

## :two_hearts:Dependencies

- torch 1.8.0
- python 3.8
- skimage 0.19.2
- numpy 1.23.1
- opencv-python 4.6.0
- timm 0.6.11
- tqdm

## :sunglasses:	Play with Demos

1. Download the [model checkpoints](https://drive.google.com/drive/folders/16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o?usp=sharing) and put the ```ckpt``` folder into the root dir.
2. Run the following commands to generate 2x and Nx (arbitrary) frame interpolation demos:

```shell
python demo_2x.py        # for 2x interpolation
python demo_Nx.py --n 8  # for 8x interpolation
```

By running above commands, you should get the follow examples by default:

<p float="left">
  <img src=figs/out_2x.gif width=340 />
  <img src=figs/out_Nx.gif width=340 /> 
</p>

## :sparkles:	Training for Fixed-timestep Interpolation

1. Download [Vimeo90K dataset](http://toflow.csail.mit.edu/)
2. Run the following command at the root dir:

```shell
  python -m torch.distributed.launch --nproc_per_node=4 train.py --world_size 4 --batch_size 8 --data_path **YOUR_VIMEO_DATASET_PATH** 
```

The default training setting is *Ours*. If you want train *Ours_small* or your own model, you can modify the ```MODEL_CONFIG``` in  ```config.py```.

## :runner:	Evaluation

1. Download the dataset you need:

   * [Vimeo90K dataset](http://toflow.csail.mit.edu/)
   * [UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow)
   * [Xiph dataset](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark_xiph.py)
   * [MiddleBury OTHER dataset](https://vision.middlebury.edu/flow/data/)
   * [SNU-FILM dataset](https://myungsub.github.io/CAIN/)
   * [HD dataset](https://github.com/baowenbo/MEMC-Net)
   * [X4K1000FPS dataset](https://www.dropbox.com/sh/duisote638etlv2/AABJw5Vygk94AWjGM4Se0Goza?dl=0)

2. Download the [model checkpoints](https://drive.google.com/drive/folders/16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o?usp=sharing) and put the ```ckpt``` folder into the root dir.

For 2x interpolation benchmarks:

```shell
python benchmark/**dataset**.py --model **model[ours/ours_small]** --path /where/is/your/**dataset**
```

For 4x interpolation benchmarks:

```shell
python benchmark/**dataset**.py --model **model[ours_t/ours_small_t]** --path /where/is/your/dataset
```

You can also test the inference time of our methods on the $H\times W$ image with the following command:

```shell
python benchmark/TimeTest.py --model **model[ours/ours_small]** --H **SIZE** --W **SIZE**
```

## :muscle:	Citation

If you think this project is helpful in your research or for application, please feel free to leave a star⭐️ and cite our paper:

```
@inproceedings{Zhang2023ExtractingMA,
  title={Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation},
  author={Guozhen Zhang and Yuhan Zhu and Hongya Wang and Youxin Chen and Gangshan Wu and Limin Wang},
  year={2023}
}
```

## :heartpulse:	License and Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE), [PvT](https://github.com/whai362/PVT), [IFRNet](https://github.com/ltkong218/IFRNet), [Swin](https://github.com/microsoft/Swin-Transformer) and [HRFormer](https://github.com/HRNet/HRFormer). Please also follow their licenses. Thanks for their awesome works.
