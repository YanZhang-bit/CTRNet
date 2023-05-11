# CTRNet
The code of "Content Temporal Relation Network for Temporal Action Proposal Generation"
# Contents
----

* [Usage Guide](#usage-guide)
   * [Prerequisites](#prerequisites)
   * [Code and Data Preparation](#code-and-data-preparation)
      * [Get the code](#get-the-code)
      * [Download Features](#download-features)
   * [Training CTRNet](#training-ctrnet)
   * [Testing CTRNet](#testing-ctrnet)


# Usage Guide

## Prerequisites

The training and testing in CTRNet is reimplemented in PyTorch for the ease of use. 

- [PyTorch 1.8.0][pytorch]
- [Fastai 1.0.61][fastai]

## Code and Data Preparation

### Get the code

Clone this repo with git, **please remember to use --recursive**

```bash
git clone --recursive https://github.com/YanZhang-bit/CTRNet
```
### Download Features
THUMOS14 feature can be download [here](https://drive.google.com/drive/folders/10PGPMJ9JaTZ18uakPgl58nu7yuKo8M_k?usp=sharing).

ActivityNet-1.3 feature can be download [here](https://drive.google.com/file/d/1VW8px1Nz9A17i0wMVUfxh6YsPCLVqL-S/view?usp=sharing). And people need rescaled the feature length of all videos to same length 100.
## Training CTRNet
* To train our CTRNet on the THUMOS14 dataset
 ```shell script
cd Thumos
python fastai_train.py
```
* To train our CTRNet on the ActivityNet-1.3 dataset
 ```shell script
cd ActivityNet
python fastai_train_content.py
```
## Testing CTRNet
* To test our CTRNet on the THUMOS14 dataset

   We provide the final model for THUMOS14 dataset [Baidu Cloud](https://pan.baidu.com/s/1PhvVBfb09MpKjiNrmQXfSg) (password: eyy6) 
 ```shell script
cd Thumos
python fastai_inference.py --checkpoint best_model/final_model.pth
python postprocess.py
python eval.py
```
* To test our CTRNet on the ActivityNet-1.3 dataset

  We provide the final model for ActivityNet-1.3 dataset [Baidu Cloud](https://pan.baidu.com/s/1pixDG7hDPBubbHeTD0L7UQ)) (password: 3qtc)
 ```shell script
cd ActivityNet
python inference_content.py --mode inference --checkpoint best_model/final_model.pth
```
