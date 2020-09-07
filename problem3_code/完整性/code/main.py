# 头部文件引用
from __future__ import print_function
import os
import sys
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from datetime import timedelta
from sklearn import metrics
from model_generate.TextCNN.cnn_model import TCNNConfig, TextCNN
from model_generate.TextCNN.cnews_loader import read_vocab, read_category, batch_iter, begin_process_file,end_process_file, build_vocab

# 获得需要的评分的数据
filename1 = "./data/predict.txt"
test_dir = "./data/pre_data.txt"

# 获得首位部模型的vocab
vocab_dir1 = './model_generate/begin_judge/data/cnews/vocab.txt'
vocab_dir2 = './model_generate/end_judge/data/cnews/vocab.txt'

# 获得首尾部模型的参数
begin_save_path =  './model_generate/begin_judge/checkpoints/textcnn/best_validation'  # begining最佳验证结果保存路径
end_save_path = './model_generate/end_judge/checkpoints/textcnn/best_validation'  # ending最佳验证结果保存路径


def data_preprocess():
    # 数据预处理函数
    lines = open(filename1,"r",encoding="utf-8-sig").readlines()
    f1 = open(test_dir,"w",encoding="utf-8-sig")
    for line in lines:
        str = "".join(line.split())+"\n"
        f1.write(str)
    f1.close()

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    # 数据加载
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def get_begin_score():
    # 首部模型评分函数
    start_time = time.time()
    x_test, y_test = begin_process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=begin_save_path)  # 读取保存的模型

    loss_test, acc_test = evaluate(session, x_test, y_test)

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果

    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 生成预测结果
    return y_pred_cls

def get_end_score():
    # 尾部模型评分函数
    start_time = time.time()
    x_test, y_test = end_process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=end_save_path)  # 读取保存的模型

    loss_test, acc_test = evaluate(session, x_test, y_test)

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果

    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
    # 生成预测结果
    return y_pred_cls

def draw_pie_chart(score):
    # 绘制饼状图
    plt.rcParams['font.sans-serif']=['SimHei'] # 解决中文乱码问题
    great =0
    fine = 0
    bad = 0
    for d in score:
        if d==2:
            great = great+1
        elif d==1:
            fine = fine+1
        else:
            bad = bad+1
    plt.figure(figsize=(6,9))
    labels = ['很完整','比较完整','不完整']
    colors = ['lightskyblue','yellowgreen','red']
    values = [great,fine,bad]
#    plt.title("文本完整性评分饼状图",fontsize=25)
    plt.pie(
            values,
            explode=(0.02,0,0),
            labels=labels,
            colors=colors,
            startangle = 90,
            shadow=False,
            autopct='%3.2f%%',
            pctdistance = 0.6
            )
    plt.axis('equal')
    plt.legend()
    plt.savefig('../pic/完整性饼状图.png')
    plt.show()


if __name__ == "__main__":
    print('启动数据完整性评分：\n')
    
    data_preprocess() #数据预处理
    score = []
    config = TCNNConfig()  # 获得TCNNConfig设置，TCNNConfig表示CNN配置参数
    categories, cat_to_id = read_category()  # read_category()获取目录，cat_to_id 标签:序号的字典
    
    print("开始计算首部完整性评分...")   
    words, word_to_id = read_vocab(vocab_dir1)  # 将词汇表的各个单词编号
    config.vocab_size = len(words)  # 更新词汇表长度
    model = TextCNN(config)  # 构建CNN模型，很重要
    bs = get_begin_score()

    tf.reset_default_graph()  # 在进入尾部模型时需要重置tensorflow图表

    print("开始计算尾部完整性评分...")
    words, word_to_id = read_vocab(vocab_dir2)  # 将词汇表的各个单词编号
    config.vocab_size = len(words)
    model = TextCNN(config)
    es = get_end_score()

    score = bs+es # 获得最后预测分数

    # 保存评分结果
    lines = open("./data/pre_data.txt","r",encoding="utf-8-sig").readlines()
    f2 = open("./data/result.txt","w",encoding="utf-8-sig")
    for i in range(len(score)):
        string = str(score[i]) + "\t" + lines[i]
        # print(string)
        f2.write(string)

    #绘制结果图表
    draw_pie_chart(score)
    print("首尾部完整性评分完毕...")