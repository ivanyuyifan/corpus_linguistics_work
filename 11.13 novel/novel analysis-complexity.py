import math
import os
import nltk
import pandas as pd
import re

result_word_complexity = []

# 定义数据清理的函数，这边把[]及其中间的内容用正则表达式清理了
def preprocess_article(processed_content):
    processed_content = re.sub(r'\[.*?\]', '', processed_content)
    
    return processed_content

#读取文件的函数
def file_read(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

folder_path = '/Users/fafaya/Desktop/corpus_linguistics_work/11.13 novel/novel text'
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(folder_path, filename)
        
        # Read and clean the content of the novel
        novel_content = file_read(filepath)
        cleaned_novel_contents = preprocess_article(novel_content)
        
        # Tokenize words and get POS tags
        words = nltk.word_tokenize(cleaned_novel_contents)
        
        # Initialize metrics variables
        token_num = 0
        words_dedup = []  # Store unique words
        
        # Process each word in the novel
        for w in words:
            w = w.lower()
            if w != 'SYM':
                token_num += 1
                if w not in words_dedup:
                    words_dedup.append(w)
        
        # Calculate the unique word count (type number)
        type_num = len(words_dedup)  # Number of unique words
        
        # Calculate various TTR metrics
        TTR = type_num / token_num if token_num > 0 else 0
        RTTR = type_num / math.sqrt(token_num) if token_num > 0 else 0
        CTTR = type_num / math.sqrt(2 * token_num) if token_num > 0 else 0
        LogTTR = math.log10(type_num) if type_num > 0 else 0
        
        # Calculate Uber metric with checks to avoid division by zero
        if token_num > 0 and type_num > 0:
            try:
                Uber = math.log10(token_num) * math.log10(token_num) / math.log10(token_num / type_num)
            except ZeroDivisionError:
                Uber = 0  # In case of division by zero, set Uber to 0
        else:
            Uber = 0
        
        # Store the results for the current novel
        metrics = {
            '小说名': filename,
            '不重复的单词数': type_num,
            'TTR（type/token）': TTR,
            'RTTR': RTTR,
            'CTTR': CTTR,
            'LogTTR': LogTTR,
            'Uber': Uber,
        }
        result_word_complexity.append(metrics)
result_df = pd.DataFrame(result_word_complexity)
result_df.to_excel('/Users/fafaya/Desktop/corpus_linguistics_work/11.13 novel/word complexity analysis.xlsx', index=False)





    