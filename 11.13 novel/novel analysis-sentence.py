import os
import re
import nltk
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

# 定义数据清理的函数，这边把[]及其中间的内容用正则表达式清理了
def preprocess_article(processed_content):
    processed_content = re.sub(r'\[.*?\]', '', processed_content)
    
    return processed_content

# 读取文件的函数
def file_read(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


### 以下统计每本小说的句子情况
def sent_count(filename, cleaned_novel):
    # 初始化每本小说的统计数据
    metrics = {
        '小说名': filename,
        '句子总数': 0,
        '小句总数': 0,
        '平均句子长度': 0,
        '平均小句长度': 0,
        '陈述句数量': 0,
        '疑问句数量': 0,
        '感叹句数量': 0,
        '其他句子数量': 0
    }
    sentences = nltk.sent_tokenize(cleaned_novel)
    metrics['句子总数'] = len(sentences)

    # 统计每个句子的类别以及相关信息
    for sent in sentences:
        if sent.endswith('.'):
            metrics['陈述句数量'] += 1
        elif sent.endswith('?'):
            metrics['疑问句数量'] += 1
        elif sent.endswith('!'):
            metrics['感叹句数量'] += 1
        else:
            metrics['其他句子数量'] += 1

        # 统计小句，以逗号分割
        small_sents = sent.split(',')
        metrics['小句总数'] += len(small_sents)
        words = nltk.word_tokenize(sent)
        metrics['平均句子长度'] += len(words)

        # 进入小句
        for sub_sent in small_sents:
            sub_words = nltk.word_tokenize(sub_sent)  # 对小句进行分词
            metrics['平均小句长度'] += len(sub_words)

    # 计算平均句子长度和平均小句长度
    if metrics['句子总数'] > 0:
        metrics['平均句子长度'] = metrics['平均句子长度'] / metrics['句子总数']
    if metrics['小句总数'] > 0:
        metrics['平均小句长度'] = metrics['平均小句长度'] / metrics['小句总数']
    
    return metrics

### 以下统计小说中的从句数量
# 加载spaCy模型
def clause_count(text):
    doc = nlp(text)
    
    # 初始化统计数据
    clause_metrics = {
        '定语从句': 0,
        '宾语从句': 0,
        '状语从句': 0,
        '名词性从句': 0
    }
    
    # 遍历每个句子，识别其中的从句
    for sent in doc.sents:
        for token in sent:
            # 定语从句通常由关系代词引导，如 who, which, that 等
            if token.dep_ == 'relcl':
                clause_metrics['定语从句'] += 1
            # 宾语从句通常由连词引导，如 that, if, whether 等
            elif token.dep_ == 'ccomp':
                clause_metrics['宾语从句'] += 1
            # 状语从句通常由从属连词引导，如 because, if, when 等
            elif token.dep_ == 'advcl':
                clause_metrics['状语从句'] += 1
            # 名词性从句通常由 that, whether, if 等引导
            elif token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                clause_metrics['名词性从句'] += 1

    return clause_metrics

### 以下统计每本小说的依存分析结果
def spacy_parsing(content):
    metrics = {
        '逻辑关系结构': 0,
        '连词结构': 0,
        '限定词结构': 0,
        '名词性主语结构': 0,
        '介词宾语结构': 0
    }
    logical_connectives = ['and',  'or', 'but', 'so', 'because', 'although', 'however']

    doc = nlp(content)
    for token in doc:
        # 统计逻辑关系结构
        if token.text.lower() in logical_connectives:
            metrics['逻辑关系结构'] += 1
        # 统计连词结构
        if token.dep_ == 'cc':
            metrics['连词结构'] += 1
        # 统计限定词结构
        elif token.dep_ == 'det':
            metrics['限定词结构'] += 1
        # 统计名词性主语结构
        elif token.dep_ == 'nsubj':
            metrics['名词性主语结构'] += 1
        # 统计介词宾语结构
        elif token.dep_ == 'pobj':
            metrics['介词宾语结构'] += 1

    return metrics


# 在主程序中执行
def process_novels(folder_path):
    record_sent_count = []  # 存放每本小说结果的空列表

    # 遍历文件夹中的所有文本文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            # 读取文件内容并清理
            novel_content = file_read(filepath)
            cleaned_novel = preprocess_article(novel_content)

            # 统计句子和从句数量
            sent_metrics = sent_count(filename, cleaned_novel)  # 句子统计
            clause_metrics = clause_count(novel_content)          # 从句统计
            spacy_metrics = spacy_parsing(novel_content)

            # 合并结果
            sent_metrics.update(clause_metrics)
            sent_metrics.update(spacy_metrics)
            record_sent_count.append(sent_metrics)

    # 将所有小说的统计数据转换为 DataFrame
    record_sent_count_df = pd.DataFrame(record_sent_count)
    return record_sent_count_df


if __name__ == '__main__':
    # 设置文件夹路径
    folder_path = '/Users/fafaya/Desktop/corpus_linguistics_work/11.13 novel/novel text'

    # 获取所有小说的句子统计数据
    result_df = process_novels(folder_path)

    # 将结果保存为 Excel 文件
    output_file = '/Users/fafaya/Desktop/corpus_linguistics_work/11.13 novel/novel analysis on sentence.xlsx'
    result_df.to_excel(output_file, index=False)
