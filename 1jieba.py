import jieba
from os import path  #用来获取文档的路径
import jieba.analyse as anls
from PIL import Image
import numpy as  np
import matplotlib.pyplot as plt
#词云生成工具
from wordcloud import WordCloud,ImageColorGenerator
#需要对中文进行处理
import matplotlib.font_manager as fm

#背景图
bg=np.array(Image.open("boy.png"))

#获取当前的项目文件加的路径
d=path.dirname(__file__)
#读取停用词表
stopwords = [line.strip() for line in open('cn_stopwords.txt', encoding='UTF-8').readlines()]  
#读取要分析的文本
text_path="answers.txt"
#读取要分析的文本，读取格式
text=open(path.join(d,text_path),encoding="utf8").read()

text_split = jieba.cut(text)  # 未去掉停用词的分词结果   list类型

#去掉停用词的分词结果  list类型
text_split_no = []
for word in text_split:
    if word not in stopwords:
        text_split_no.append(word)
#print(text_split_no)
fW = open('fencioutput.txt','w',encoding = 'UTF-8')
fW.write(' '.join(text_split_no))
fW.close()

text_split_no_str =' '.join(text_split_no)  #list类型分为str

with open('fencioutput.txt',"r",encoding = 'UTF-8') as r:
                lines =r.readlines()
with open('fencioutput.txt',"w",encoding = 'UTF-8') as w:
                for line in lines:
                       if len(line) > 2:
                           w.write(line)

fW = open('fencioutput1.txt','w',encoding = 'UTF-8')
fW.write(' '.join(text_split_no))
fW.close()

text_split_no_str =' '.join(text_split_no)  #list类型分为str

#基于tf-idf提取关键词
print("基于TF-IDF提取关键词结果：")
keywords = []
for x, w in anls.extract_tags(text_split_no_str, topK=200, withWeight=True):
    keywords.append(x)   #前200关键词组成的list
keywords = ' '.join(keywords)   #转为str
print(keywords)
print("基于词频统计结果")
txt = open("fencioutput1.txt", "r", encoding="UTF-8").read()
words = jieba.cut(txt)
counts = {}
for word in words:
    if len(word) == 1:
        continue
    else:
        rword = word
    counts[rword] = counts.get(rword, 0) + 1
items = list(counts.items())
items.sort(key=lambda x:x[1], reverse=True)
for i in range(33):
    word, count=items[i]
    print((word),count)
#生成
wc=WordCloud(
    background_color="white",
    max_words=200,
    mask=bg,            #设置词云形状
    max_font_size=60,
    scale=16,
    random_state=42,
    font_path='simhei.ttf'   #中文处理，用系统自带的字体
    ).generate(keywords)
#为图片设置字体
my_font=fm.FontProperties(fname='simhei.ttf.ttf')
#产生背景图片，基于彩色图像的颜色生成器
image_colors=ImageColorGenerator(bg)
#开始画图
plt.imshow(wc,interpolation="bilinear")
#为云图去掉坐标轴
plt.axis("off")
#画云图，显示
#plt.figure()
plt.show()
#为背景图去掉坐标轴
plt.axis("off")
plt.imshow(bg,cmap=plt.cm.gray)
#plt.show()

#保存云图
wc.to_file("ciyun.png")
print("词云图片已保存")
