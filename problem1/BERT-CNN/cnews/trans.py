import csv
import matplotlib.pyplot as plt
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np


def trans(input_file, output_file):

    labels = []
    features = []

    with open(input_file, 'r') as fh:
        lines = fh.readlines()

        print(len(lines))

        for line in lines:
            label, feature = line.split('\t')
            feature = feature.strip('\n')
            labels.append(label)
            features.append(feature)
    
    with open(output_file, 'w') as f:
        out_writer = csv.writer(f, delimiter='\t')
        out_writer.writerow(['sentence', 'label'])
        print("labels:{}".format(len(labels)))
        for i in range(len(labels)):
            out_writer.writerow([features[i], labels[i]])

        
def analysis(filename, type, tokenizer):
    text_lens = []

    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for line in reader:
            text = tokenizer.tokenize(line[0])
            text_lens.append(len(text))

    x = list(range(len(text_lens)))

    plt.plot(x, text_lens, label="text length")

    # 设置坐标轴范围
    plt.ylim((0, 10000))

    # 设置坐标轴刻度
    y_ticks = np.arange(0, 10000, 500)
    plt.yticks(y_ticks)

    plt.title(type)
    plt.ylabel("text length")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # trans("train.txt", "train.tsv")
    # trans("val.txt", "dev.tsv")
    # trans("test.txt", "test.tsv")

    tokenizer = BertTokenizer.from_pretrained(
        "/home/songyingxin/datasets/pytorch-bert/vocabs/bert-base-chinese-vocab.txt", do_lower_case=True)

    analysis("train.tsv", "train", tokenizer)
    # analysis("dev.tsv", "dev", tokenizer)
    # analysis("test.tsv", "test", tokenizer)

