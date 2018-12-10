# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:04:06 2018

@author: dell
"""

# 获取名称
data = open("dinos.txt", "r").read()

# 转化为小写字符
data = data.lower()

# 转化为无序且不重复的元素列表
chars = list(set(data))

# 获取大小信息
data_size, vocab_size = len(data), len(chars)

print(chars)
print("共计有%d个字符，唯一字符有%d个"%(data_size,vocab_size))

char_to_ix = {ch:i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i:ch for i, ch in enumerate(sorted(chars))}

with open("dinos.txt") as f:
        examples = f.readlines()
examples = [x.lower().strip() for x in examples]


index = 1 % len(examples)
X = [None] + [char_to_ix[ch] for ch in examples[index]] 
Y = X[1:] + [char_to_ix["\n"]]