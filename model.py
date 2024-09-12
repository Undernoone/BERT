import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# Transformer Embedding is a combination of word embedding and positional encoding
# BERT's Embedding has an additional Segment Embedding feature

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

# 使用正则表达式去除标点符号和换行符，并转换成小写
# re.sub() 方法将[.,!?\'"-]替换为空格
# split() 方法用于分割字符串，默认是以空格为分隔符
# First, we create spaces, then we find spaces to split the sentences.
sentences = re.sub(r'[.,!?\'"-]', '', text.lower()).split('\n')
# Make sentences into a long string
# Get the unique words in the text,because we need to get the index of each word.
sentences_join = " ".join(sentences)
word_list = list(set(sentences_join.split()))

word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
# Create index to word dictionary
# 例如，如果 word2idx 中有 {'hello': 4}，则 idx2word 会有 {4: 'hello'}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {index:word for word, index in word2idx.items()}
vocab_size = len(word2idx)

token_list = []
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    '''
    sentence: original sentence
    sentence.split(): split sentence into one word per element
    arr: list of word index
    '''
    token_list.append(arr)

max_len = 30
batch_size = 6
max_pred = 5
n_layers = 6
n_heads = 12
d_model = 768
d_ff = 768 * 4 # fnn_size usually is 4 times of d_model
d_k = d_v = 64
n_segments = 2

def make_data():
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        #random.randrange(start, end, step=1):从[start, end)区间中以step为步长随机选取一个数 = for i in range(start, end, step)
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences)) # sample random index in sentences
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        #   [CLS] Nice meet you too. [SEP] How are you today? [SEP]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        #   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15))) # 15 % of tokens in one sentence, at most 5
        # document position of masked tokens
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']
                          ]

        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        #选前n_pred个单词进行mask
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos) # document input_ids index
            masked_tokens.append(input_ids[pos]) # document input_ids value
            if random() < 0.8:
                #[0, 0.8) make mask
                input_ids[pos] = word2idx['[MASK]']

                #[0.8, 0.9) don't make mask
            elif random() > 0.9:
                # [0.9, 1.0) replace with random word, but didn't use 'CLS', 'SEP', 'PAD'
                # random index in vocabulary
                index = randint(0, vocab_size - 1)
                while index < 4: # if index < 4, replace with random word again to avoid 'CLS', 'SEP', 'PAD'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index

        # Padding
        n_pad = max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        '''
        pred = (batch_size, max_pred, d_model)-->(batch_size, max_pred, vocab_size)-->(batch_size*max_pred, vocab_size)
        masked_tokens.shape = (batch_size, max_pred)--->(batch_size*max_pred, )
        nn.CrossEntropyLoss(pred, masked_tokens)

        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        (batch_size, maxlen, d_model)--->(batch_size, max_pred, d_model):torch.gather
        '''

        # Make sure positive and negative are balanced
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
    return batch

batch = make_data()
"""
Example of a batch[0]:
[
[1, 35, 3, 3, 26, 37, 34, 36, 2, 35, 19, 12, 26, 37, 34, 36, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Word index (0 is padding)
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Different segment index (0 is segment A, 1 is segment B)
[19, 12, 0, 0, 0], # Before masking index
[2, 3, 0, 0, 0], # Mask index
False # IsNext or NotNext
] 
"""
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids = torch.LongTensor(input_ids)
segment_ids = torch.LongTensor(segment_ids)
masked_tokens = torch.LongTensor(masked_tokens)
masked_pos = torch.LongTensor(masked_pos)
isNext = torch.LongTensor(isNext)

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[idx]
dataset = MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext)
loader = Data.DataLoader(dataset, batch_size, True)

def get_attn_pad_mask(seq_q, seq_k):
    """
    Generate the attention mask for padding token.
    Let attention doesn't consider padding token, so we can mask padding token with 0.
    """
    batch_size, seq_len = seq_q.size()
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.seg_embedding = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x, seg):
        seq_len = x.size(1)
        batch_size = x.size(0)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
        embedding = self.tok_embedding(x) + self.pos_embedding(pos) + self.seg_embedding(seg)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.dot_attention = ScaledDotProductAttention()#
        self.linear = nn.Linear(d_v * n_heads, d_model)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context = self.dot_attention(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        return self.norm(output + residual)

class PositionWiseFFN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(PositionWiseFFN, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PositionWiseFFN()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        embed_weight = self.embedding.tok_embedding.weight
        self.fc2 = nn.Linear(d_model, vocab_size)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        h_pooled = self.fc(output[:, 0])
        logits_clsf = self.classifier(h_pooled)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.activ2(self.linear(h_masked))
        logits_lm = self.fc2(h_masked)
        return logits_lm, logits_clsf

import matplotlib.pyplot as plt

model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)

# 保存损失值
losses = []

for epoch in range(50):
    epoch_loss = 0
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))
        loss_clsf = criterion(logits_clsf, isNext)
        loss = loss_lm + loss_clsf
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_epoch_loss = epoch_loss / len(loader)
    losses.append(avg_epoch_loss)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1:04d}, loss = {avg_epoch_loss:.6f}')

input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[1]
print(text)
print('================================')
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), \
                               torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ',True if logits_clsf else False)

plt.plot(range(1, 51), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()


















