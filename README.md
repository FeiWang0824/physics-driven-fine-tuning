# physics-driven-fine-tuning

demo code for [Single-pixel imaging using physics enhanced deep learning](https://opg.optica.org/prj/fulltext.cfm?uri=prj-10-1-104).


## Abstract
Here we propose a physics-enhanced deep learning approach for SPI. By blending a physics-informed layer and a model-driven fine-tuning process, we show that the proposed approach is generalizable for image reconstruction. We implement the proposed method in an in-house SPI system and an outdoor single-pixel LiDAR system. The proposed method establishes a bridge between data-driven and model-driven algorithms, allowing one to impose both data and physics priors for inverse problem solvers in computational imaging, ranging from remote sensing to microscopy.

## Overview
![avatar](https://pcsdata.baidu.com/thumbnail/9c71ab99em26e606ee363c2de2f1ca1b?fid=2326259770-16051585-437861194953908&rt=pr&sign=FDTAER-yUdy3dSFZ0SVxtzShv1zcMqd-n3mixc0UMQQ59hYlfpOqXmaFO9w%3D&expires=2h&chkv=0&chkbd=0&chkpc=&dp-logid=8810054304714794540&dp-callid=0&time=1665194400&bus_no=26&size=c1600_u1600&quality=100&vuk=-&ft=video "Left: Schematic diagram of the physics enhanced deep learning approach for SPI. (a) The physics-informed DNN. (b) The SPI system. (c) The model-driven fine-tuning process. Right:Experimental results: images of the badge of our institute reconstructed by (a) HSI with $\beta$ = 100% (it serves as the ground truth), (b) HSI with $\beta$ = 6.25%, (c) DCAN, (d) TVAL3, (e) fine-tuning with random initialization, (f) DGI with learned patterns, (g) physics-informed DNN, and (h) the fine-tuning process.")

## Results
![avatar](https://opg.optica.org/getImage.cfm?img=QC5mdWxsLHByai0xMC0xLTEwNC1nMDA3&article=prj-10-1-104-g007 "Experimental results for single-pixel LiDAR. (a) Schematic diagram of the single-pixel LiDAR system. (b) Satellite image of our experiment scenario. The inset in the top left is the target imaged by a telescope, whereas the one in the bottom right is one of the echoed light signals. (c) Six typical 2D depth slices of the 3D object reconstructed by DGI with the learned patterns illumination, GISC [5], and the proposed fine-tuning method. (d) 3D images of the object reconstructed by the three aforementioned methods.")

## How to use
### Required packages
*conda env create -f environment.yml*

### Pretrained models
Two models trained on CelebA (128 $\times$ 128) and stl10 (64 $\times$ 64) are avaliable \[[Download pretrain models](https://drive.google.com/file/d/1AKmTzAoQA1MHlzpgzy955XJD3UfruBH4/view?usp=sharing)]. 

The total number of sampling patterns are both 1024. 

One can obtained a new pretrained model by using *pretrain.y*.

### Simulations
Run **gen_simulation_data.py** to generate data required for simulation.

Run **finetune.py** for reconstruction.

### Experiment
Using optimized patterns (part of the pretrained model) to encode objects to obtain experimental data.

Run **finetune.py** for reconstruction.

Here we provide one of experiment data (**SIOM_exp.mat**) for a quick demo. 

## Citation
Fei Wang, Chenglong Wang, Chenjin Deng, Shensheng Han and Guohai Situ. Single-pixel imaging using physics enhanced deep learning. *Photon. Res.* **10**, 1 (2022).

@article{Wang:PR22,
author = {Fei Wang and Chenglong Wang and Chenjin Deng and Shensheng Han and Guohai Situ},
title = {Single-pixel imaging using physics enhanced deep learning},
journal = {Photon. Res.},
number = {1},
pages = {104--110},
volume = {10},
month = {Jan},
year = {2022},
doi = {10.1364/PRJ.440123}}


