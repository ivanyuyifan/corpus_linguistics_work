import jieba

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:  # r是读取的意思
        return file.read()
    
text = read_file('/Users/fafaya/Desktop/语料库/语料库语言学/10.30class_test.txt')

# 使用jieba分词
words = jieba.lcut(text)

# 将分词结果写入文件
output_file_path = '/Users/fafaya/Desktop/语料库/语料库语言学/10.30class_test_output.txt'
with open(output_file_path, mode='w', encoding='utf-8') as output_file:
    output_file.write(" ".join(words))  # 将列表转换为字符串

print("分词结果已写入文件")
