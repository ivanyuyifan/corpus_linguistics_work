import jieba
import gensim
import pandas as pd
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel
import re
import pyLDAvis.gensim_models
import pyLDAvis
import json
import numpy as np

# 数据预处理函数
def clean_text(text):
    # 去除非中文字符（包括英文、数字和标点）
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)
    # 去除多余空格
    text = re.sub(r"\s+", " ", text)
    return text

# 使用for循环选择合适的主题数
def compute_coherence_values(corpus, dictionary, words, min_topics, max_topics):
    coherence_values = []
    for num_topics in range(min_topics, max_topics + 1):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=30, random_state=42)
        coherence_model = gensim.models.coherencemodel.CoherenceModel(model=model, texts=[words], dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
    return coherence_values

def main():
    # 1. 数据预处理
    # 加载中文停用词
    stopwords_path = "/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/hit_stopwords.txt"
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())

    # 加载语料库
    corpus_path = "/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/一带一路_绿色（语料全）.txt"
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 清理文本
    cleaned_text = clean_text(text)

    # 使用jieba分词
    words = jieba.cut(cleaned_text)
    words = [word for word in words if len(word) > 1 and word not in stopwords]  # 去除停用词和单字词

    # 2. 构建词袋模型（Bag of Words）
    # 创建词典
    dictionary = corpora.Dictionary([words])

    # 转换为词袋向量
    corpus = [dictionary.doc2bow(text) for text in [words]]

    # 3. 使用for循环选择合适的主题数
    coherence_values = compute_coherence_values(corpus, dictionary, words, 2, 50)  # 假设主题数从2到10
    x = range(2, 51)

    # 绘制一致性得分图
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Scores for Various Numbers of Topics")
    plt.show()

    # 根据一致性得分选择最佳主题数，例如选择一致性得分最高的主题数
    optimal_topics = coherence_values.index(max(coherence_values)) + 2
    print(f"最佳主题数：{optimal_topics}")

    # 4. LDA建模
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_topics, passes=30, random_state=42)

    # 5. 可视化LDA建模结果
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

    # 保存结果为html格式
    output_html_path = "/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/lda_visualization1205.html"
    pyLDAvis.save_html(vis, output_html_path)


    print(f"LDA可视化已保存为：{output_html_path}")

# 确保代码在主程序中运行
if __name__ == '__main__':
    main()
