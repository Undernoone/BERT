import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# Transformer_Embedding = Embedding + Positional_Encoding
# BERT_Embedding = TokenEmbedding + Positional_Encoding + Segment_Embedding +

text = (
    "Hello,how are you?I am Coder729\n"
    "Hello,Coder729,My name is Wang,nice to meet you!\n"
    "Nice to meet you,too!How are you today?\n"
    "Great!My baseball team won the championship.\n"
    "Oh,that's good.I'm glad to hear that.Wang!\n"
    "Thankyou,Coder729!\n"
)

# text = (
#     'abc+'
#     '!#$$'
# )
# print(text);exit()

sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n') # filter '.', ',', '?', '!'
# print(sentences);exit()
sentences_join = " ".join(sentences)
word_list = list(set(sentences_join.split()))
print(word_list);exit()

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





































if __name__ == "__main__":
    maxLength = 30

    pass

text = {
    "Hello,how are you?I am Coder729\n"
    "Hello,Coder729,My name is Wang,nice to meet you!\n"
    "Nice to meet you,too!How are you today?\n"
    "Great!My baseball team won the championship.\n"
    "Oh,that's good.I'm glad to hear that.Wang!\n"
    "Thankyou,Coder729!\n"
}

sentences = re.sub("[.,!?;]", "", text.lower().split("\n"))
word_list = list(set("".join(sentences).split()))
word_dict = {"[PAD]":0,'[CLS]':1,'[SEP]':2,'[Mask]':3}
for i,word in enumerate(word_list):
    word_dict[word] = i+4

