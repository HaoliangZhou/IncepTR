# IncepTR

### IncepTR: Micro-Expression Recognition Integrating Inception-CBAM and Vision Transformer ([Paper](https://link.springer.com/article/10.1007/s00530-023-01164-0))<br>
Haoliang Zhou, Shucheng Huang, and Xuqiao Xu<br> 

##

### Abstract <br>
Micro-Expressions (MEs) are the instantaneous and subtle facial movement that conveys crucial emotional information. However, traditional neural networks face difficulties in accurately capturing the delicate features of MEs due to the limited amount of available data. To address this issue, a dual-branch attention network is proposed for ME recognition, called IncepTR, which can capture attention-aware local and global representations. The network takes optical flow features as input and performs feature extraction using a dual-branch network. First, the Inception model based on the Convolutional Block Attention Module (CBAM) attention mechanism is maintained for multi-scale local feature extraction. Second, the Vision Transformer (ViT) is employed to capture subtle motion features and robustly model global relationships among multiple local patches. Additionally, to enhance the rich relationships between different local patches in ViT, Multi-head Self-Attention Dropping (MSAD) is introduced to drop an attention map randomly, effectively preventing overfitting to specific regions. Finally, the two types of features could be used to learn ME representations effectively through similarity comparison and feature fusion. With such combination, the model is forced to capture the most discriminative multi-scale local and global features while reducing the influence of affective-irrelevant features. Extensive experiments show that the proposed IncepTR achieves UF1 and UAR of 0.753 and 0.746 on the composite dataset MEGC2019-CD, demonstrating better or competitive performance compared to existing state-of-the-art methods for ME recognition.

<p align="center">
<img src="https://github.com/HaoliangZhou/IncepTR/blob/master/fig.png" width=100% height=100% 
class="center">
</p>

### Data preparation
Following [Dual-ATME](https://github.com/HaoliangZhou/Dual-ATME#data-preparation) and [RCN](https://github.com/xiazhaoqiang/ParameterFreeRCNs-MicroExpressionRec/blob/main/PrepareData_LOSO_CD.py), the data lists are reorganized as follow:

```
data/
├─ MEGC2019/
│  ├─ v_cde_flow/
│  │  ├─ 006_test.txt
│  │  ├─ 006_train.txt
│  │  ├─ 007_test.txt
│  │  ├─ ...
│  │  ├─ sub26_train.txt
│  │  ├─ subName.txt
```
1. There are 3 columns in each txt file: 
```
/home/user/data/samm/flow/006_006_1_2_006_05588-006_05562_flow.png 0 1
```
In this example, the first column is the path of the optical flow image for a particular ME sample, the second column is the label (0-2 for three emotions), and the third column is the database type (1-3 for three databases).

2. There are 68 raws in _subName.txt_: 
```
006
...
037
s01
...
s20
sub01
...
sub26
```
Represents ME samples divided by MEGC2019, as described in [here](https://facial-micro-expressiongc.github.io/MEGC2019/) ahd [here](https://facial-micro-expressiongc.github.io/MEGC2019/images/MEGC2019%20Recognition%20Challenge.pdf).


### Citation <br>
If you find this repo useful for your research, please consider citing the paper
```
@Article
}
```

