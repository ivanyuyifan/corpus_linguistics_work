from ltp import LTP,StnSplit
from collections import Counter
import os

# 文件路径
file_path = "/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/一带一路_绿色（语料全）.txt"

#加载哈工大停用词表
stopwords_file = open('/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/四个中文停用词表/hit_stopwords.txt','r',encoding='utf-8').read()
zh_stopwords = set(stopwords_file.split('\n'))  # 使用 set 提升查找效率

# 读取文件内容
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    content = content.replace('\n','')
    content = content.replace(' ','')
    content = content.replace('\t','')
    content = content.strip()

# 初始化LTP实例
ltp = LTP()  
# 对文本进行分句（可选，根据语料情况）
sentences = StnSplit().split(content)

# 对每个句子进行分词
tokenized_words = []
for sent in sentences:
    output = ltp.pipeline(sent,tasks=['cws'])
    words = output.cws
    tokenized_words.extend([word for word in words if word not in zh_stopwords and word.strip()])  # 过滤停用词和空白词


# 定义函数生成n-grams列表
def generate_ngrams(tokens, n=2):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

# 生成二元组和三元组
bigrams = generate_ngrams(tokenized_words, n=2)
trigrams = generate_ngrams(tokenized_words, n=3)

# 统计N-gram的频率
bigram_counts = Counter(bigrams)
trigram_counts = Counter(trigrams)

# 打印出现频率最高的前30个2-gram和3-gram
print("Top 30 Bigrams:")
for bg, freq in bigram_counts.most_common(30):
    print(f"{bg} : {freq}")

print("\nTop 30 Trigrams:")
for tg, freq in trigram_counts.most_common(30):
    print(f"{tg} : {freq}")