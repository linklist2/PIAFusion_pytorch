# PIAFusion_pytorch

The Chinese version of the README.md could be found in [README_中文.md](https://github.com/linklist2/PIAFusion_pytorch/blob/master/README_%E4%B8%AD%E6%96%87.md).

If you think this project is helpful to you, please click the star button in the upper right corner!

This is **unofficial** pytorch implementation of “PIAFusion: A progressive infrared and visible image fusion network based on illumination aware” by [linklist2](https://github.com/linklist2).

Official **tensorflow implementation** of this paper and more details could be found in **[PIAFusion](https://github.com/Linfeng-Tang/PIAFusion)**.

Our experiments were tested on both win10 1080Ti and Ubuntu 20.04 3090 and found no bugs.

It is worth noting that the loss function in the version implemented by pytorch is consistent with the loss function in the paper, and may be slightly different from the tensorflow version.

Many thanks to the author for his help during code debugging！

## Update
 - Picture in picture(**[plotRegionZoom.py](https://github.com/linklist2/PIAFusion_pytorch/blob/master/utils/plotRegionZoom.py)**):
  
   ![image](utils/multiregion.bmp)

## Recommended Environment(win10 1080Ti)

 - torch 1.9.0 
 - tqdm 4.64.0  
 - numpy 1.22.4
 - opencv-python 4.5.5.64

More detailed environment requirements can be found in ```requirements.txt```. 

## To Training

 ### 1. Convert data_illum.h5 file to image form
The dataset for training the illumination-aware sub-network can be download from [data_illum.h5](https://pan.baidu.com/s/19Xbg3bWcMo600zZe7exnVg?pwd=PIAF).

Create ```datasets``` folder in this project, and then move the downloaded h5 file into it.

For easy viewing of images and debugging code, plz run the following code:
```shell
python trans_illum_data.py --h5_path 'datasets/data_illum.h5' --cls_root_path 'datasets/cls_dataset'
```
The converted directory format is as follows:
```shell
 cls_dataset/
 ├── day
 │   ├── day_0.png
 │   ├── day_1.png
 │   ├── ......
 ├── night
 │   ├── night_0.png
 │   ├── night_1.png
 │   ├── ......
```

 ### 2. Convert data_MSRS.h5 file to image form
The dataset for training the illumination-aware fusion network can be download from [data_MSRS.h5](https://pan.baidu.com/s/1cO_wn2DOpiKLjHPaM1xZYQ?pwd=PIAF).

For easy viewing of images and debugging code, plz download the file and run the following code:
```shell
python trans_msrs_data.py --h5_path 'datasets/data_MSRS.h5' --msrs_root_path 'datasets/msrs_train'
```

The converted directory format is as follows:
```shell
 msrs_train/
 ├── Inf
 │   ├── 0.png
 │   ├── 1.png
 │   ├── ......
 ├── Vis
 │   ├── 0.png
 │   ├── 1.png
 │   ├── ......
```

If the link given above has expired, you can download the dataset [here](https://pan.baidu.com/s/18XjhLlzr_t9Y1sDYudJHww?pwd=u1tt). 


### 3. Training the Illumination-Aware Sub-Network
```shell
python train_illum_cls.py --dataset_path 'datasets/cls_dataset' --epochs 100 --save_path 'pretrained' --batch_size 128 --lr 0.001
```
Then the weights of the best classification model can be found in [pretrained/best_cls.pth](https://github.com/linklist2/PIAFusion_pytorch/blob/master/pretrained/best_cls.pth), The test accuracy of the best model is around 98%.

### 4. Training the Illmination-Aware Fusion Network
```shell
python train_fusion_model.py --dataset_path 'datasets/msrs_train' --epochs 30 --cls_pretrained 'pretrained/best_cls.pth' --batch_size 128 --lr 0.001 --loss_weight '[3, 7, 50]' --save_path 'pretrained'
```
The values in **loss_weight** correspond to **loss_illum**, **loss_aux**, **gradinet_loss** respectively.


## To Testing
### 1. The MSRS Dataset
```shell
python test_fusion_model.py --h5_path 'test_data/MSRS' --save_path 'results/fusion' --fusion_pretrained 'pretrained/fusion_model_epoch_29.pth'
```

The fusion result can be found in the directory corresponding to the ```save_path``` parameter.

It can be observed that the results is not particularly ideal and needs to be further adjusted.

**Note: The directory structure of the test dataset should be the same as that of the training dataset, as follows**:



```shell
 MSRS/
 ├── Inf
 │   ├── 00537D.png
 │   ├── ......
 ├── Vis
 │   ├── 00537D.png
 │   ├── ......
```

# TODO

 - [ ] Test The RoadScene Dataset
 - [ ] Test The TNO Dataset  
 - [ ] Adjust the loss factor parameter
 - [ ] Modify the loss function


## If this work is helpful to you, please cite it as：
```
@article{Tang2022PIAFusion,
  title={PIAFusion: A progressive infrared and visible image fusion network based on illumination aware},
  author={Tang, Linfeng and Yuan, Jiteng and Zhang, Hao and Jiang, Xingyu and Ma, Jiayi},
  journal={Information Fusion},
  volume = {83-84},
  pages = {79-92},
  year = {2022},
  issn = {1566-2535},
  publisher={Elsevier}
}
```
