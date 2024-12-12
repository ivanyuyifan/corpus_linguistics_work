import jieba
import gensim
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel
import re
import pyLDAvis.gensim_models
import pyLDAvis
from wordcloud import WordCloud

# 数据预处理函数
def clean_text(text):
    # 去除非中文字符（包括英文、数字和标点）
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)
    # 去除多余空格
    text = re.sub(r"\s+", " ", text)
    return text

def plot_combined_wordcloud(model, num_topics, num_words=50):
    # 初始化一个字典来存储所有主题的词和它们的权重
    combined_words = {}

    # 遍历所有主题
    for t in range(num_topics):
        topic_words = model.show_topic(t, num_words)
        for word, weight in topic_words:
            if word in combined_words:
                combined_words[word] += weight  # 如果词已经存在，累加权重
            else:
                combined_words[word] = weight  # 如果词不存在，初始化权重

    # 生成词云
    wordcloud = WordCloud(font_path="/Users/fafaya/Library/Fonts/SimHei.ttf", width=800, height=600, background_color='white').generate_from_frequencies(combined_words)
    
    # 显示词云
    plt.figure(figsize=(10, 8))
    plt.title(f"A General Theme WordCloud", fontsize=16)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


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

    # 3. LDA建模，使用23个主题
    optimal_topics = 23
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_topics, passes=30, random_state=42)

    # 4. 计算困惑度
    perplexity = lda_model.log_perplexity(corpus)
    print(f"困惑度: {perplexity}")

    # 5. 计算一致性得分
    coherence_model = gensim.models.coherencemodel.CoherenceModel(model=lda_model, texts=[words], dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"一致性得分: {coherence_score}")

    # 6. 可视化LDA建模结果
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

    # 保存结果为html格式
    output_html_path = "/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/lda_visualization_23topicnums.html"
    pyLDAvis.save_html(vis, output_html_path)

    # 7. 生成并显示综合词云图（结合所有主题）
    plot_combined_wordcloud(lda_model, num_topics=optimal_topics)

    print(f"LDA可视化已保存为：{output_html_path}")

# 确保代码在主程序中运行
if __name__ == '__main__':
    main()
