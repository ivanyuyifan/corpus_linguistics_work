import re

# 读取原始文件内容
with open("/Users/fafaya/Desktop/corpus_linguistics_work/11.20 语料库与文学、体裁/Romeo and Juliet.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 定义角色列表
characters = [
    "CHORUS", "SAMPSON", "GREGORY", "ABRAM", "BALTHASAR", "BENVOLIO",
    "TYBALT", "CAPULET", "LADY CAPULET", "MONTAGUE", "LADY MONTAGUE",
    "PRINCE", "ROMEO", "JULIET", "NURSE", "MERCUTIO", "PARIS", "FRIAR LAWRENCE"
]

# 构建正则表达式：匹配角色名所在的行及其后紧跟的台词
pattern = r"^(?P<character>" + "|".join(characters) + r")\.\s*\n(?P<dialogue>(?:[^\n]+\n)+)"

# 添加标注标签
def tag_dialogue(match):
    character = match.group("character").strip()  # 角色名
    dialogue = match.group("dialogue").strip()  # 台词
    return f"<{character}>\n{dialogue}\n</{character}>"

# 替换匹配的台词为带标注的格式
tagged_text = re.sub(pattern, tag_dialogue, text, flags=re.MULTILINE)

# 保存标注后的文本文件
with open("/Users/fafaya/Desktop/corpus_linguistics_work/11.20 语料库与文学、体裁/Tagged_Romeo_and_Juliet.txt", "w", encoding="utf-8") as file:
    file.write(tagged_text)

print("标注完成！标注后的文本已保存为 'Tagged_Romeo_and_Juliet.txt'")
