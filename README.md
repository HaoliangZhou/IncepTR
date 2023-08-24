# IncepTR

Code of the paper "IncepTR: Micro-Expression Recognition Integrating Inception-CBAM and Vision Transformer" ([Paper](xxx))<br>
Haoliang Zhou, Shucheng Huang, and Xuqiao Xu<br> 

### Abstract <br>
xxx

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

