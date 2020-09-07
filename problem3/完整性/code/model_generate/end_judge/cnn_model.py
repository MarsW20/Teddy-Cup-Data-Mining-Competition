# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 2  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 300  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-4  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 100  # 总迭代轮次

    print_per_batch = 10  # 每多少轮输出一次结果
    save_per_batch = 5  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        # 每个元素被保留的概率，那么 keep_prob:1就是所有元素全部保留的意思。一般在大量数据训练时，为了防止过拟合，添加Dropout层，设置一个0~1之间的小数
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer，important
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer全局池化
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
            
            with tf.name_scope("bias"):
                cnn_b = tf.get_default_graph().get_tensor_by_name("conv/bias:0")
                tf.summary.histogram("cnn_bias", cnn_b)

            with tf.name_scope("weight"):
                cnn_w = tf.get_default_graph().get_tensor_by_name("conv/kernel:0")
                tf.summary.histogram("cnn_weights", cnn_w)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)  
            
            with tf.name_scope("bias"):    
                fc1_b = tf.get_default_graph().get_tensor_by_name("fc1/bias:0")
                tf.summary.histogram("fc1_bias",fc1_b)

            with tf.name_scope("weight"):
                fc1_w = tf.get_default_graph().get_tensor_by_name("fc1/kernel:0")
                tf.summary.histogram("fc1_weights",fc1_w)  

        with tf.name_scope("classify"):
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            
            with tf.name_scope("bias"):    
                fc2_b = tf.get_default_graph().get_tensor_by_name("fc2/bias:0")
                tf.summary.histogram("fc2_bias",fc2_b)

            with tf.name_scope("weight"):
                fc2_w = tf.get_default_graph().get_tensor_by_name("fc2/kernel:0")
                tf.summary.histogram("fc2_weights",fc2_w) 

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



