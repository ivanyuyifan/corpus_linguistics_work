import os
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd

punctuation =  r"""!"#$%&'’()*+,-./:;<=>?@[\]^_`{|}~“”--"""
en_stopwords = stopwords.words('english')

pos_dict = {'名词': ['NN','NNS','NNP','NNPS'],
 '形容词': ['JJ','JJR','JJS'],
 '动词': ['VB','VBD','VBG','VBN','VBP','VBZ','MD'],
 '代词': ['PRP','WP','WP$'],
 '数词': ['CD'],
 '副词': ['RB','RBR','RBS','WRB'],
 '介词': ['IN'],
 '连词': ['CC','IN'],
 '冠词': ['DT'],
 '叹词': ['UH'],
 }

# 定义数据清理的函数
def preprocess_article(processed_content):
    processed_content = re.sub(r'\[.*?\]', '', processed_content)
    
    return processed_content

#读取文件的函数
def file_read(filepath):
    file = open(filepath, 'r', encoding='utf-8')
    content = file.read()

    return content

folder_path = '/Users/fafaya/Desktop/corpus_linguistics_work/11.13 novel/novel text'

wordlength_result = []#记录词语特征的列表

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(folder_path, filename)
        novel_content = file_read(filepath)
        cleaned_novel = preprocess_article(novel_content)

        words = nltk.word_tokenize(cleaned_novel)

###以下内容统计词数和词长（统计和分布）###

        word_length = {}
        total_num = 0
        for w in words:
            if w not in punctuation:
                total_num += 1
                length = len(list(w))
                word_length[length] = word_length.get(length, 0) + 1

        word_length_sorted = dict(sorted(word_length.items(), key=lambda x:x[1]))

        word_len_distribution = {}#词长字典
        for k, v in word_length_sorted.items():
            word_len_distribution[k] = v / total_num

        wordlength_metrics = {
            '小说名' : filename,
            '总词数': total_num,
            '词长统计': word_length_sorted,
            '词长分布': word_len_distribution
}
        
        wordlength_result.append(wordlength_metrics)
wordlength_result = pd.DataFrame(wordlength_result)           

###以下内容统计词性###
pos_result = []
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(folder_path, filename)
        novel_content = file_read(filepath)
        cleaned_novel = preprocess_article(novel_content)

        words = nltk.word_tokenize(cleaned_novel)
        word_pos = nltk.pos_tag(words)

        #最终输出的结果就是metrics
        metrics = { '小说名': filename,
        '总词数': 0,
        '名词': 0,
        '形容词': 0,
        '动词': 0,
        '代词': 0,
        '数词': 0,
        '副词': 0,
        '介词': 0,
        '连词': 0,
        '冠词': 0,
        '叹词': 0,
        '实词': 0,
        '虚词': 0,
        '词汇词': 0
        }
        word_frequency = {}#词频字典

        for w_p in word_pos:
            w = w_p[0].lower()
            p = w_p[1]

            if w not in punctuation:
                metrics['总词数'] = metrics.get('总词数') + 1

                if w not in en_stopwords:
                    word_frequency[w] = word_frequency.get(w, 0) + 1

                for k, v in pos_dict.items():
                    if p in v:
                        p = k

                pos_tags = list(pos_dict.keys())
                if p in pos_tags:
                    metrics[p] = metrics.get(p) + 1

                if p in ['名词','形容词','动词','代词','数词','副词']:
                    metrics['实词'] = metrics.get('实词') + 1
                elif p in ['介词','连词','冠词','叹词']:
                    metrics['虚词'] = metrics.get('虚词') + 1

                if p in ['名词','形容词','动词','副词']:
                    metrics['词汇词'] = metrics.get('词汇词') + 1


        metrics['词汇密度'] = metrics['词汇词'] / metrics['总词数']
        word_frequency_sorted = dict(sorted(word_frequency.items(), key=lambda x:x[1], reverse = True))
        metrics['（去除停用词）词频'] = word_frequency_sorted

        pos_result.append(metrics)

pos_result = pd.DataFrame(pos_result)
      
 #把两个结果合并起来
merged_df = pd.merge(wordlength_result, pos_result, on='小说名', how='outer')

# 保留一个 '总词数' 列，这里直接选择 '总词数_x'
if '总词数_x' in merged_df.columns:
    merged_df['总词数'] = merged_df['总词数_x']
    merged_df = merged_df.drop(['总词数_x', '总词数_y'], axis=1)

# 重新调整列的位置，使得 '总词数' 紧跟在 '小说名' 后面
columns = ['小说名', '总词数'] + [col for col in merged_df.columns if col not in ['小说名', '总词数']]
merged_df = merged_df[columns]

# 输出到 Excel 文件
output_path = '/Users/fafaya/Desktop/corpus_linguistics_work/11.13 novel/novel analysis on word.xlsx'
merged_df.to_excel(output_path, index=False)

print("数据已经合并并保存到 Excel 文件。")
       
##还需要做可视化内容：词长统计和词长分布的折线图，词频的柱状图（top10-20）   
##也可以考虑做词汇丰富度



        

        
        
