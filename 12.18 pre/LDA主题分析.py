import os
import jieba
import pandas as pd
from tqdm import tqdm
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import warnings
import re

# 指定字体文件的路径
font_path = '/Users/fafaya/Library/Fonts/SimHei.ttf'

# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = font_path
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
warnings.filterwarnings("ignore")  # 过滤掉所有的警告信息

# 读取单个文档内容
def get_data(file_path):
    data = []  # 用于记录文档内容
    
    # 读取指定文件内容
    with open(file_path, mode='r', encoding='utf-8') as f:
        content = f.read()
    
    # 进行文本预处理
    content = content.replace('\n', '').replace(' ', '').replace('\u3000', '')  # 剔除换行符，半角空格，全角空格
    processed_content = preprocessing(content)  # 调用自定义的preprocessing函数进行处理
    
    # 将处理后的内容存储在列表中
    data.append(processed_content) 

    return data  # 返回文档内容

def preprocessing(content):
    stopwords_path = '/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/hit_stopwords.txt'
    stopwords = open(stopwords_path, mode='r', encoding='utf-8').read().split('\n')  # 读取停用词表

    key_words = []
    words = jieba.cut(content)  # 使用jieba分词
    for w in words:
        # 过滤条件：停用词、数字、包含%符号的词汇、纯英文单词、长度小于等于1的词汇、由数字或非字母字符组成的词汇
        if (w not in stopwords and  # 不是停用词
            not re.match(r'^\d+(\.\d+)?$', w) and  # 不是数字（整数或小数）
            '%' not in w and  # 不包含%
            not re.match(r'^[a-zA-Z]+$', w) and  # 不是纯英文单词
            len(w) > 1 and  # 长度大于1
            not re.match(r'^[\W\d_]+$', w)):  # 不是仅包含数字、标点或非字母字符
            key_words.append(w)

    cleaned_keywords = clean_keywords(key_words)  # 假设 clean_keywords 是你之前定义的清洗词汇的函数

    return cleaned_keywords  # 返回非停用词的词汇列表

# 清理关键词，包括替换特殊字符和多余空格
def clean_keywords(keywords):
    # 替换所有 \xa0 为普通空格
    keywords = [keyword.replace('\xa0', ' ') for keyword in keywords]
    
    # 移除多余的省略号字符，替换为单一空格
    keywords = [keyword.replace('…', ' ') for keyword in keywords]
    
    # 去除每个元素前后空格并合并多余的空格
    keywords = [re.sub(r'\s+', ' ', keyword.strip()) for keyword in keywords]
    
    return keywords

# 将词袋结果可视化成dataframe，便于观察词汇稀疏性
def doc2bow_visualization(dictionary, corpus_doc2bow):
    token2id = dictionary.token2id
    # 翻转字典的键和值，获得 id-词汇 的字典
    id2token = {value: key for key, value in token2id.items()}

    corpus_doc2bow_df = pd.DataFrame()  # 创建空的dataframe
    for document_id, words in enumerate(corpus_doc2bow):
        for word_id, value in words:
            word = id2token.get(word_id)
            corpus_doc2bow_df.at[document_id, word] = value
    corpus_doc2bow_df.fillna(0, inplace=True)  # 将空值补0

    save_to_excel_in_sheets(corpus_doc2bow_df, '/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/词袋结果可视化.xlsx')

# 将TF-IDF结果可视化成dataframe
def tfidf_visualization(dictionary, corpus_tfidf):
    token2id = dictionary.token2id
    # 翻转字典的键和值，获得 id-词汇 的字典
    id2token = {value: key for key, value in token2id.items()}

    corpus_tfidf_df = pd.DataFrame()  # 创建空的dataframe
    for document_id, words in enumerate(corpus_tfidf):
        for word_id, value in words:
            word = id2token.get(word_id)
            corpus_tfidf_df.at[document_id, word] = value
    corpus_tfidf_df.fillna(0, inplace=True)  # 将空值补0

    save_to_excel_in_sheets(corpus_tfidf_df, '/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/TFIDF结果可视化.xlsx')

def save_to_excel_in_sheets(df, filename, max_columns=16384):
    # 计算需要多少个工作表
    num_sheets = (df.shape[1] // max_columns) + 1

    with pd.ExcelWriter(filename) as writer:
        for i in range(num_sheets):
            start_col = i * max_columns
            end_col = min((i + 1) * max_columns, df.shape[1])
            sheet_data = df.iloc[:, start_col:end_col]  # 获取每个工作表的子集
            sheet_data.to_excel(writer, sheet_name=f'Sheet{i+1}', index=False)


# 进入主函数，执行所有步骤
if __name__ == '__main__':
    # 获取文档内容
    file_path = '/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/一带一路_绿色（语料全）.txt'  # 这里指定文件路径
    data = get_data(file_path)  # 调用修改后的get_data，传入文件路径

    # 将data中的所有词汇提取出来，用列表存放
    all_key_words = data

     # 使用gensim库构建词典，并将所有词汇转换为词袋结果
    dictionary = corpora.Dictionary(all_key_words)  # 构建词典
    corpus_doc2bow = [dictionary.doc2bow(w) for w in all_key_words]  # 转换为doc2bow格式

    # 可视化词袋结果
    doc2bow_visualization(dictionary, corpus_doc2bow)

    # 转换为TF-IDF结果
    tfidf_model = models.TfidfModel(corpus_doc2bow)
    corpus_tfidf = tfidf_model[corpus_doc2bow]

    # 可视化TF-IDF结果
    tfidf_visualization(dictionary, corpus_tfidf)

    # LDA建模，指定主题个数为10
    num_topics = 10
    lda_model = LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=1000, random_state=1)

    # 使用pyLDAvis可视化
    lda_visualization = pyLDAvis.gensim.prepare(lda_model, corpus_tfidf, dictionary)
    pyLDAvis.save_html(lda_visualization, '/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/lda_visualization.html')

    # 计算困惑度
    perplexity = lda_model.log_perplexity(corpus_tfidf)
    print('困惑度: ', perplexity)

    # 计算一致性
    lda_cm = CoherenceModel(model=lda_model, texts=all_key_words, dictionary=dictionary, coherence='c_v')
    coherence = lda_cm.get_coherence()
    print('一致性: ', coherence)
### 至此，我们已经完成了指定主题个数为10的LDA建模。但未必个数为10时，LDA模型是合理的。
    ### 为了找到最合理的LDA模型，我们可以通过for循环，改变个数的值，找到一致性最高的情况，从而确定最合理的LDA模型。

    ### 不指定主题个数，绘制一致性曲线，自动寻找最佳的主题个数
    topic_range = range(1, 50)  # 规定主题个数的范围为1到9
    coherence_result = []  # 记录一致性
    for i in tqdm(topic_range, desc='LDA建模中...'):
        lda_model = LdaModel(corpus_tfidf, num_topics=i, id2word=dictionary, passes=30, random_state=1)
        lda_cm = CoherenceModel(model=lda_model, texts=all_key_words, dictionary=dictionary, coherence='c_v')
        coherence_result.append(lda_cm.get_coherence())

    ### 绘制一致性曲线
    x = topic_range  # x轴是主题个数的范围
    y = coherence_result  # y轴是一致性
    plt.plot(x, y)
    plt.xlabel('主题个数')
    plt.ylabel('一致性')
    plt.title('主题-一致性曲线')
    plt.grid(False)  # 取消网格背景
    plt.savefig('/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/主题-一致性曲线.png')
    plt.show()

    
