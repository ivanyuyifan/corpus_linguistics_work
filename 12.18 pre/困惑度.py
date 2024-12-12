import jieba
import gensim
import pandas as pd
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel
import re
import pyLDAvis.gensim_models
import pyLDAvis
import numpy as np

# 数据预处理函数
def clean_text(text):
    # 去除非中文字符（包括英文、数字和标点）
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)
    # 去除多余空格
    text = re.sub(r"\s+", " ", text)
    return text

# 计算一致性和困惑度的函数
def compute_metrics(corpus, dictionary, words, min_topics, max_topics):
    coherence_values = []
    perplexity_values = []
    for num_topics in range(min_topics, max_topics + 1):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=30, random_state=42)
        
        # 计算一致性值
        coherence_model = gensim.models.coherencemodel.CoherenceModel(model=model, texts=[words], dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
        
        # 计算困惑度值
        perplexity = model.log_perplexity(corpus)
        perplexity_values.append(perplexity)
        
    return coherence_values, perplexity_values

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

    # 3. 使用for循环计算一致性和困惑度
    min_topics, max_topics = 2, 50
    coherence_values, perplexity_values = compute_metrics(corpus, dictionary, words, min_topics, max_topics)
    
    # 绘制一致性和困惑度图
    x = range(min_topics, max_topics + 1)
    
    # 一致性
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, coherence_values, marker='o')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Scores for Various Numbers of Topics")
    
    # 困惑度
    plt.subplot(1, 2, 2)
    plt.plot(x, perplexity_values, marker='o')
    plt.xlabel("Number of Topics")
    plt.ylabel("Log Perplexity")
    plt.title("Perplexity for Various Numbers of Topics")
    
    plt.tight_layout()
    plt.show()

    # 打印一致性和困惑度的具体值
    for i, (coh, perp) in enumerate(zip(coherence_values, perplexity_values), start=2):
        print(f"主题数: {i}, 一致性得分: {coh:.4f}, 困惑度: {perp:.4f}")

# 自动选择：找一致性得分最高的点，同时困惑度平稳下降
    optimal_topics = np.argmax(coherence_values) + 2
    print(f"最佳主题数为: {optimal_topics}")

    # 综合考虑一致性和困惑度，选择最佳主题数
    optimal_topics = x[np.argmax(coherence_values)]  # 一致性得分最高的位置
    print(f"最佳主题数（基于一致性得分）：{optimal_topics}")

    # 4. LDA建模
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_topics, passes=30, random_state=42)

    # 5. 可视化LDA建模结果
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

    # 保存结果为html格式
    output_html_path = "/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/lda_visualization_with_metrics.html"
    pyLDAvis.save_html(vis, output_html_path)

    print(f"LDA可视化已保存为：{output_html_path}")

# 确保代码在主程序中运行
if __name__ == '__main__':
    main()
