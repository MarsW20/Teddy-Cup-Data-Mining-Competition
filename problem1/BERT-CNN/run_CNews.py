# coding=utf-8
from main import main
from BertCNN import args
from datetime import datetime
from logging_py import *

if __name__ == "__main__":

    model_name = "BertCNN"
    label_list = ['城乡建设', '环境保护', '交通运输', '教育文体', '劳动和社会保障', '商贸旅游', '卫生计生']
    data_dir = "cnews"
    output_dir = "cnews_output/"
    cache_dir = "cnews_cache/"
    log_dir = "cnews_log/"

    model_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    init_logger("logs/{}.log".format(model_path))

    bert_vocab_file = "pretrained_bert/vocab.txt"  # 需改
    bert_model_dir = "pretrained_bert"
    
    config = args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir)
    main(config, config.save_name, label_list,logger,True)



