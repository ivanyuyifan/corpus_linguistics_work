import os
import re
import pandas as pd
from tqdm import tqdm
import time
from ltp import LTP,StnSplit
ltp = LTP() #全局作用，整份代码的任何地方都能够调用这个ltp

###读取语料并进行预处理
filepath = '/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/一带一路_绿色（语料全）.txt'
with open(filepath, 'r') as f:
    content = f.read()
    content = content.replace('\n','').replace(' ','')
    content = content.replace('\u3000','')
    content = content.replace('\t','')
    content = content.strip()

###统计句子指标
record_sent_count = []
metrics = {
        '句子总数': 0,
        '小句总数': 0,
        '平均句子长度': 0,
        '平均小句长度': 0,
        '陈述句数量': 0,
        '疑问句数量': 0,
        '感叹句数量': 0,
        '其他句子数量': 0
            }
sentences = StnSplit().split(content)
metrics['句子总数'] += len(sentences)

#进入每个句子
###########
for sent in sentences:
    if sent.endswith('。'):
        metrics['陈述句数量'] += 1
    elif sent.endswith('？'):
        metrics['疑问句数量'] += 1
    elif sent.endswith('！'):
        metrics['感叹句数量'] += 1
    else:
         metrics['其他句子数量'] += 1

    output = ltp.pipeline(sent,tasks=['cws'])
    words = output.cws
    metrics['平均句子长度'] += len(words)

    sub_sents = re.split('，|；',sent)
    metrics['小句总数'] += len(sub_sents)

    #进入每个小句
    for sub_s in sub_sents:
        sub_output = ltp.pipeline(sub_s,tasks=['cws'])
        sub_words = sub_output.cws
        metrics['平均小句长度'] += len(sub_words)
###########
metrics['平均句子长度'] /= metrics['句子总数']
metrics['平均小句长度'] /= metrics['小句总数']
record_sent_count.append(metrics)
record_sent_count = pd.DataFrame(record_sent_count)
record_sent_count.to_excel('/Users/fafaya/Desktop/corpus_linguistics_work/12.18 pre/result/句子层面统计result.xlsx', index=False)
print('结果已成功保存')