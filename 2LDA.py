import numpy as np
from gensim import corpora, models


if __name__ == '__main__':
    # 读入文本数据
    f = open('fencioutput.txt', encoding='utf-8')  # 输入已经预处理后的文本
    texts = [[word for word in line.split()] for line in f]
    f.close()
    M = len(texts)
    print('文本数目：%d 个' % M)

    # 建立词典
    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)
    print('词的个数：%d 个' % V)

    # 计算文本向量g
    corpus = [dictionary.doc2bow(text) for text in texts]  # 每个text对应的稀疏向量

    # 计算文档TF-IDF
    corpus_tfidf = models.TfidfModel(corpus)[corpus]

    # LDA模型拟合
    num_topics = 10  # 定义主题数
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha=0.01, eta=0.01, minimum_probability=0.001,
                          update_every=1, chunksize=100, passes=1)

    # 所有文档的主题
    doc_topic = [a for a in lda[corpus_tfidf]]
    print('Document-Topic:')
    print(doc_topic)

    # 打印文档的主题分布
    num_show_topic = 5  # 每个文档显示前几个主题
    print('文档的主题分布：')
    doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    idx = np.arange(M)  # M为文本个数，生成从0开始到M-1的文本数组
    for i in idx:
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:, 1])
        topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]  # 按照概率大小进行降序排列
        print('第%d个文档的前%d个主题：' % (i, num_show_topic))
        print(topic_idx)
        print(topic_distribute[topic_idx])

    # 每个主题的词分布
    num_show_term = 15  # 每个主题显示几个词
    for topic_id in range(num_topics):
        print('主题#%d：\t' % topic_id)
        term_distribute_all = lda.get_topic_terms(topicid=topic_id)  # 所有词的词分布
        term_distribute = term_distribute_all[:num_show_term]  # 只显示前几个词
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        print('词：', end="")
        for t in term_id:
            print(dictionary.id2token[t], end=' ')
        print('概率：', end="")
        print(term_distribute[:, 1])

    # 将主题-词写入一个文档 topword.txt，每个主题显示20个词
    with open('ldatopic.txt', 'w', encoding='utf-8') as tw:
        for topic_id in range(num_topics):
            term_distribute_all = lda.get_topic_terms(topicid=topic_id, topn=20)
            term_distribute = np.array(term_distribute_all)
            term_id = term_distribute[:, 0].astype(np.int)
            for t in term_id:
                tw.write(dictionary.id2token[t] + " ")
            tw.write("\n")

