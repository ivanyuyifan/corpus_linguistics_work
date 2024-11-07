import re

sentence = 'France won the FIFA World Cup 2018'
sentence = ''
output = re.match(r'.+',sentence) #一个可以匹配任何字符串的正则表达式

print(output)
