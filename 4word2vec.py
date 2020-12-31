# -*- coding: utf-8 -*-
 
import logging
 
from gensim import models
from gensim.models import word2vec
 
 
def main(): 
    # word2vec模型读取
    # model = word2vec.Word2Vec.load("your_model_name")
 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = models.Word2Vec.load('corpusSegDone_1.model')
 
    print("提供 3 种测试模式\n")
    print("输入一个词，则去寻找前20个该词的相似词")
    print("输入两个词，则去计算两个词的余弦相似度")
    print("输入三个词，进行类比推理")
 
    while True:
        try:
            query = input('')
            q_list = query.split()
 
            if len(q_list) == 1:
                print("相似词前 100 排序")
                res = model.most_similar(q_list[0], topn=20)
                for item in res:
                    print(item[0] + "," + str(item[1]))
 
            elif len(q_list) == 2:
                print("计算 Cosine 相似度")
                res = model.similarity(q_list[0], q_list[1])
                print(res)
            else:
                #q0相比q2，就如同q1相比于q3.q0-q1+q2=q3
                print("%s之于%s，如%s之于" % (q_list[0], q_list[2], q_list[1]))
                res = model.most_similar([q_list[0], q_list[1]], [q_list[2]], topn=20)
                for item in res:
                    print(item[0] + "," + str(item[1]))
            print("----------------------------")
        except Exception as e:
            print(repr(e))
 
 
if __name__ == "__main__":
    main()
