import os
import re
import nltk

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

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(folder_path, filename)
        novel_content = file_read(filepath)
        cleaned_novel = preprocess_article(novel_content)

        sentences = nltk.sent_tokenize(cleaned_novel)