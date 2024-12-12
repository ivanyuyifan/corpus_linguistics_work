import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pandas as pd
from ltp import LTP,StnSplit
ltp = LTP()
from wordcloud import WordCloud #生成词云
from matplotlib import pyplot as plt

filepath = '/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/一带一路_绿色（语料全）.txt'
with open(filepath, 'r') as f:
    content = f.read()
    
    content = content.replace('\n','')
    content = content.replace(' ','')
    content = content.replace('\t','')
    content = content.strip()

result = []
word_length = {} #词长记录
total_num = 0 #总词数
### 进入每个句子
sentences = StnSplit().split(content)
for sent in sentences:
    output = ltp.pipeline(sent,tasks=['cws','pos'])
    words = output.cws
    pos = output.pos
    ### 进入每个单词
    for w,p in zip(words,pos):
        if p != 'wp':
            total_num += 1
            length = len(w)
            word_length[length] = word_length.get(length,0) + 1
            ############# 执行完毕后，就相当与整个content就处理完毕了

word_length_sorted = dict(sorted(word_length.items(),key=lambda x:x[1],reverse=True))
#### 将word_length_sorted计算成新的一个分布字典
word_len_distribution = {}
for k,v in word_length_sorted.items():
    word_len_distribution[k] = v / total_num
            ########################

metrics = {
                '总词数': total_num,
                '词长统计': word_length_sorted,
                '词长分布': word_len_distribution
            }
result.append(metrics)

result = pd.DataFrame(result)
result.to_excel('/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/result/词长统计result.xlsx', sheet_name='result', index=False)
print('结果已成功输出')

# 过滤词长，保留合理范围（例如词长 <= 30）
filtered_word_lengths = {k: v for k, v in word_len_distribution.items() if k <= 32}

# 对过滤后的词长进行排序
sorted_word_lengths = sorted(filtered_word_lengths.keys())  # 从小到大排序
distribution_values = [filtered_word_lengths[k] for k in sorted_word_lengths]  # 获取对应值

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(sorted_word_lengths, distribution_values, marker='o', linestyle='-', color='b', alpha=0.7)

# 设置图表标题和轴标签
plt.title('Word Length Distribution', fontsize=16)
plt.xlabel('Word Length', fontsize=14)
plt.ylabel('Proportion', fontsize=14)

# 添加网格
plt.grid(axis='both', linestyle='--', alpha=0.7)

# 美化横坐标刻度
plt.xticks(sorted_word_lengths, fontsize=12)
plt.yticks(fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()