import pandas as pd
from ltp import LTP, StnSplit
ltp = LTP()
import pandas as pd
import math
from matplotlib import pyplot as plt

#加载哈工大停用词表
stopwords_file = open('/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/四个中文停用词表/hit_stopwords.txt','r',encoding='utf-8').read()
zh_stopwords = stopwords_file.split('\n')

filepath = '/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/一带一路_绿色（语料全）.txt'
with open(filepath, 'r') as f:
    content = f.read()
    
    content = content.replace('\n','')
    content = content.replace(' ','')
    content = content.replace('\t','')
    content = content.strip()

result_duoyangxing = []# 用于记录词汇多样性指标
# metrics用于记录词汇多样性指标
metrics = {
    '不重复的单词数': 0,
    'TTR（type/token）': 0,
    'RTTR': 0,
    'CTTR': 0,
    'LogTTR': 0,
    'Uber': 0,
}
token_num = 0
words_dedup = [] #用于记录一篇作文中不重复的单词

sentences = StnSplit().split(content)
for sent in sentences:
    output = ltp.pipeline(sent,tasks=['cws','pos'])
    words = output.cws
    pos = output.pos

    for w,p in zip(words,pos):
        if p != 'wp':
            token_num += 1 # 相当于token_num = token_num + 1

            if w not in words_dedup:
                words_dedup.append(w)

type_num = len(words_dedup)
metrics['不重复的单词数'] = type_num
metrics['TTR（type/token）'] = type_num / token_num
metrics['RTTR'] = type_num / math.sqrt(token_num)
metrics['CTTR'] = type_num / math.sqrt(2 * token_num)
metrics['LogTTR'] = math.log10(type_num) / math.log10(token_num)
metrics['Uber'] = math.log10(token_num) * math.log10(token_num) / math.log10(token_num / type_num)

result_duoyangxing.append(metrics)
result_duoyangxing = pd.DataFrame(result_duoyangxing)
result_duoyangxing.to_excel('/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/result/多样性result.xlsx',sheet_name='result',index=True)
print('多样性结果已成功输出至/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/result/多样性result.xlsx')