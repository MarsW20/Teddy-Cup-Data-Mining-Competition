import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #解决中文乱码

def paint_(y):
    cnt_1=0
    cnt_2=0
    cnt_3=0
    for i in y:
        if i<1.5:   cnt_1+=1
        elif i>1.5 and i<2.2:    cnt_2+=1
        else: cnt_3+=1

    plt.figure(figsize=(6, 9))  # 调节图形大小
    labels = [u'可解释', u'较可解释', u'不可解释']  # 定义标签
    sizes = [cnt_3, cnt_2, cnt_1]  # 每块值
    print(cnt_1,cnt_2,cnt_3)

    colors = ['red', 'yellowgreen', 'lightskyblue']  # 每块颜色定义
    explode = (0, 0, 0.02)  # 将某一块分割出来，值越大分割出的间隙越大
    patches, text1, text2 = plt.pie(sizes,
                                    explode=explode,
                                    labels=labels,
                                    colors=colors,
                                    labeldistance=1.05,  # 图例距圆心半径倍距离
                                    autopct='%3.2f%%',  # 数值保留固定小数位
                                    shadow=False,  # 无阴影设置
                                    startangle=90,  # 逆时针起始角度设置
                                    pctdistance=0.6)  # 数值距圆心半径倍数距离
    # patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部文本
    # x，y轴刻度设置一致，保证饼图为圆形
    plt.axis('equal')
    plt.legend()
    plt.show()

"""文本长度"""
length=pd.read_excel("附件4.xlsx",usecols=[5]).values.tolist()
length=[len(x[0]) for x in length]

"""相关性"""
corr=[float(line.strip()) for line in  open("../Correlation/res.txt","r").readlines()]

"""因果关系"""
relation=[float(line.strip()) for line in  open("因果关系.txt","r").readlines()]

"""entropy"""
entropy=[float(line.strip()) for line in  open("entropy.txt","r").readlines()]

print(corr[0],relation[0],entropy[0],length[0],max(corr))
score=[]
for i in range(len(corr)):
    score.append(corr[i]/max(corr)+
                 relation[i]/max(relation)+
                 entropy[i]/max(entropy)+
                 length[i]/max(length))

y=score
print(score)
with open("可解释性.txt","w") as f:
    for i in y:
        f.write(str(i)+'\n')
    f.close()

x=[i for i in range(len(y))]
plt.plot(x,y)
plt.ylabel("explaniable")
plt.title("Explaniable")
#plt.show()
paint_(score)
