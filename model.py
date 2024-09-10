import re # 正则表达式库
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# Transformer_Embedding = Embedding + Positional_Encoding
# BERT_Embedding = TokenEmbedding + Positional_Encoding + Segment_Embedding +

# Demo，So I create a simple text, rather than a real text dataset.
text = (
    'Hello, how are you? I am Coder729.\n' # Coder729
    'Hello, Coder729 My name is Wang. Nice to meet you.\n' # Wang
    'Nice meet you too. How are you today?\n' 
    'Great. My baseball team won the competition.\n' 
    'Oh Congratulations, Wang\n' 
    'Thank you Coder729\n' 
    'Where are you going today?\n' 
    'I am going shopping. What about you?\n' 
    'I am going to visit my grandmother. she is not very well'
)

# text = (
#     'abc+'
#     '!#$$'
# )
# print(text);exit()

# 使用正则表达式去除标点符号和换行符，并转换成小写
# re.sub() 方法将[.,!?\'"-]替换为空格
# split() 方法用于分割字符串，默认是以空格为分隔符
# Perfect code! First, we create space, then we find split to split the sentences.
sentences = re.sub(r'[.,!?\'"-]', '', text.lower()).split('\n')
print(sentences)
sentences_join = " ".join(sentences)
word_list = list(set(sentences_join.split()))
print(word_list)

word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {index:word for word, index in word2idx.items()}
vocab_size = len(word2idx)

token_list = []
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    # print(sentence)
    # print(sentence.split())
    # print(arr)
    # exit()
    token_list.append(arr)







































