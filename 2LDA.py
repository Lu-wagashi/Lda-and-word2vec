from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer  
import numpy as np
import lda
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
import collections
if __name__ == "__main__":
    corpus = []
    for line in open('fencioutput.txt','r',encoding='utf-8').readlines():
        corpus.append(line.strip())
    print (corpus)
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()   # 所有的特征词，即关键词
    #print (word)    
    #print(X)
    analyze = vectorizer.build_analyzer()  
    weight = X.toarray()  
    #print(weight)
    
    # 训练模型
    model = lda.LDA(n_topics = 13, n_iter =100, random_state = 1,alpha=0.03,eta=0.01)
    model.fit(np.asarray(weight))
    
    # 主题-词分布
    topic_word = model.topic_word_  #生成主题以及主题中词的分布
    #print("topic-word:\n", topic_word)
    # 计算topN关键词
    n = 10
    for i, word_weight in enumerate(topic_word):  
        #print("word_weight:\n", word_weight)
        distIndexArr = np.argsort(word_weight)
        #print("distIndexArr:\n", distIndexArr)
        topN_index = distIndexArr[:-(n+1):-1]
        #print("topN_index:\n", topN_index) # 权重最多的n个
        topN_words = np.array(word)[topN_index]    
        print(u'*Topic {}\n- {}'.format(i, ' '.join(topN_words)))  
        # 文档-主题分布
    
    doc_topic = model.doc_topic_ 
    #print("type(doc_topic): {}".format(type(doc_topic)))  
    #print("shape: {}".format(doc_topic.shape))
    label = []        
    for i in range(doc_topic.shape[0]):  
        #print(doc_topic[i])
        topic_most_pr = doc_topic[i].argmax()  
        label.append(topic_most_pr)  
        #print("doc: {} topic: {}".format(i, topic_most_pr))  
    c = collections.Counter(label)
    print(c)
