# 基于TextCNN模型的第一题求解

## 模型思想

### 模型示意图

<img src="./pic/cnn_architecture.png"  height="530" width="895">

### 模型阐述

​		我们利用基于TextCNN的模型对我们第一题的文本数据进行分类。TextCNN主要由嵌入层，卷积层，池化层以及全连接层构成。TextCNN作为CNN的变种模型，有较为成熟的结构以及众多的参考资料，该模型训练速度快，能反应文本局部特征的，由此实现文本的分类。

## 模型构建

### 运行环境

> python 3.5
>
> tensorflow 1.5
>
> sklearn
>
> numpy

### 数据预处理

​		抽取附件二中的文本中"一级标签","留言主题","留言详情"，将"留言主题作为label",将留言主题与留言详情进行合并作为content，然后乱序处理后，切割成7:2:1文本，命名为train.txt,val.txt,test.txt，并将其放置在./code/data/cnews/ 内构成我们的初始文本数据。

### 训练模型

​		于code文件夹下，在命令行输入指令：

> python run_cnn.py train

开始模型的训练。训练结果会随之输出。

<img src="./pic/train.png"  height="430" width="695">

### 测试模型结果

​		于code文件夹下，在命令行输入指令：

> python run_cnn.py test

<img src="./pic/test.png"  height="430" width="695">



