# BERT

ELMO使用了双向信息，但是网络架构太老了是RNN，我不是预测未来，我是完型填空

GPT使用了新的Transformer，但是是单向

所以我使用了双向的Transformer

## Input

### NSP二分类任务

## Abstract

BERT全名为Bidirectional Encoder Representations from Transformers。

与最近（2018）的模型不同，BERT使用的是未标注文本的深度双向考虑上下文训练模式。

因此，BERT只需要添加一个额外输出层进行微调。随实现简单但证实强大。

## Introduction

常用的预训练方法有

基于特征的方法如ELMo：通过增加预训练的语言表示来辅助特定任务。

微调方法如GPT：调整预训练好的模型参数来适应下游任务。

这两种方法都不太好，尤其是是微调方法，因为它使用的是单向语言模型，只能从左往右提取特征信息，对于双向需要上下文理解的任务处理不好，这种限制对于文本模型来说是灾难性的。

BERT改进了基于微调的方法，它使用了掩码语言模型缓解了单向语言模型的限制。

什么是掩码语言模型呢？掩码语言模型就是随机掩盖住一些输入，让机器依据上下文去预测被掩盖的部分（下文会详细讲解实现方法）。

## Related Work

讲解NLP的发展历史和常用技术

提出了LSTM和ELMo的优点和不足

### BERT

BERT框架主要技术只有预训练和微调

### Model Architecture

多层双向Transformer机制，相比单向的Transformer模型如GPT，BERT能够更好地理解上下文

**Input/Output Representations**

数据输入所谓的"句子"不一定是一个句子，可能是两个句子组合在一起，也可能是一个句子被从中间截断了，如（Python is the best language in the world，it can do anything可能会被截断为"e best language in the world，it can do anyt"）,但是无论被截断成什么样子第一个标记总是[CLS]且都会被[SEP]标记分开。并通过嵌入来区分不同的Token ，如Embedding、Seg Embedding、Pos Embedding

## Mask机制

[CLS]和[SEP]不会被mask

15%会被Mask

80%真正的mask

10%不变

10%随机替换