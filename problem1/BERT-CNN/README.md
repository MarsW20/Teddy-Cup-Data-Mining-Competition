"""
    本项目实现了BERT-CNN。
"""

"""
    #数据预处理
    7:2:1分成 train.csv,dev.csv,test.csv
    每个文件共2列，第一列是label,第二列是文本
"""

1、下载'bert-base-chinese':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt"

2、下载vocab.txt
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz"

3、将下载文件移至pretrained_bert文件夹,最后会形成如下目录
pretrained_bert:
                bert_config.json
                pytorch_model.bin
                vocab.txt
4、模型训练+预测
python run_CNews.py --max_seq_length=128 --num_train_epochs=1.0 --do_train --gpu_ids="1" --gradient_accumulation_steps=2 --print_step=500

只预测:
python run_CNews.py --max_seq_length=128 --num_train_epochs=20.0 --gpu_ids="1" --gradient_accumulation_steps=2 --print_step=500

5、tensorboard可视化训练过程
    tensorboard --logdir cnews_log


