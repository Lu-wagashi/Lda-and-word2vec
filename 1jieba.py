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

