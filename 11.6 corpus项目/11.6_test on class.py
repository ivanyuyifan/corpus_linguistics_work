import jieba
import jieba.posseg as pseg
import re
import chardet
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


filepath = '/Users/fafaya/Desktop/corpus.txt'

# 检测文件编码
with open(filepath, "rb") as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(f"检测到文件编码为：{encoding}")

# 使用检测到的编码读取文件内容，并忽略解码错误
with open(filepath, "r", encoding=encoding, errors='ignore') as file:
    content = file.read()

# 使用正则表达式去除【】及其中的内容
cleaned_content = re.sub(r'【.*?】', '', content, flags=re.S)
cleaned_content = re.sub(r'\[(.*?)\]', r'\1', cleaned_content)  # 去除 [] 保留内容

# 将清理后的内容写回文件
with open("corpus_processed.txt", "w", encoding="utf-8") as file:
    file.write(cleaned_content)

# 分词和词性标注
words_with_pos = pseg.cut(cleaned_content)
#只保留带“吃”且长度为2的词语
filtered_data = []

for word, pos in words_with_pos:
    if '吃' in word and len(word) == 2:
        position = word.index('吃')  # 确定“吃”字的位置（0表示首位，1表示末位）
        filtered_data.append([position, word, pos])

# 将结果保存到 DataFrame
df = pd.DataFrame(filtered_data, columns=["位置", "词语", "词性"])

# 将结果保存到 Excel 文件
df.to_excel("result.xlsx", index=False, sheet_name="Eat Word Analysis")
print("分析结果已保存到 result.xlsx 文件中。")

# 显著性分析：检查词性与“吃”字位置的关系
# 创建交叉表，统计不同位置的词性分布
contingency_table = pd.crosstab(df["位置"], df["词性"])

# 使用卡方检验来判断关系的显著性
chi2, p, dof, expected = chi2_contingency(contingency_table)

# 样本量
n = len(df)

print(f"卡方检验结果：chi2 = {chi2:.4f}, p = {p:.4f}")
print(p)
if p < 0.05:
    print(f"卡方检验表明“吃”字的位置与词性之间存在显著关系，χ²({dof}, N = {n}) = {chi2:.2f}, p = {p:.3f}。")
else:
    print(f"卡方检验未能表明“吃”字的位置与词性之间的显著关系，χ²({dof}, N = {n}) = {chi2:.2f}, p = {p:.3f}。")