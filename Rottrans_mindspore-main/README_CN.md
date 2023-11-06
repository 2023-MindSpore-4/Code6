# 目录

<!-- TOC -->

- [目录](#目录)
- [RotTrans描述](#RotTrans描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
- [随机情况说明](#随机情况说明)

<!-- /TOC -->

# RotTrans描述

RotTrans采用vit作为backbone，增强了对大旋转差异的鲁棒性。此外，RotTrans设计了不变性约束来建立原始特征和旋转特征之间的关系，从而实现更强的旋转不变性。因此可以更好地完成无人机视角下采集的数据集的行人重识别。

[论文](https://dl.acm.org/doi/abs/10.1145/3503161.3547799)：Chen S, Ye M, Du B. Rotation Invariant Transformer for Recognizing Object in UAVs[C]//Proceedings of the 30th ACM International Conference on Multimedia. 2022: 2565-2574.


# 数据集

使用的数据集：[PRAI-1581](https://github.com/stormyoung/PRAI-1581/blob/master/README.md)

- 数据集大小：共39461张行人矩形图像，包含1581个类，由2个大疆无人机摄像头拍摄。
- 数据集分割：按照RotTrans原论文中的分割方式，对PRAI-1581数据集进行分割。
    - 训练集
        - train: 19523张行人图像，包含781类。
    - 测试集
        - query: 4680张行人图像，包含799类。
        - gallery: 15258张行人图像，包含799类。
- 数据格式：jpg
    - 注：数据将在data_manager.py 和dataset_loader.py中处理。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)



# 脚本说明

## 脚本及样例代码

```
├── RotTrans
    ├── config					参数相关
        ├── __init__.py
        └── default.py				默认参数
    ├── model					模型相关
        ├── backbones				模型骨架
            ├── __init__.py
            └── vit_pytorch.py			vit骨架
		├── __init__.py
        └── make_model.py			搭建模型
    ├── src					其他脚本
        ├── __init__.py
        ├── customloss.py			损失函数
		├── dataset.py			创建数据集
		├── dataset_define.py		数据集定义
		├── metrics.py			模型评估指标
		├── samplers.py			采样
        └── transforms.py			数据增强
	├── utils				
        ├── __init__.py
        └── logger.py				生成日志
├── test.py					测试脚本
├── train.py					训练脚本
    └── README_CN.md


```


## 脚本参数

在config文件夹中的defaults.py中可以配置训练与评估参数，其中附有介绍参数的注释。


## 训练过程

正在改进训练过程，提高训练速度与精度。




# 随机情况说明

在MStrain.py中，设置了"set_seed"的种子。

