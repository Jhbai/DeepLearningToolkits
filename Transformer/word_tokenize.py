import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt

# 第一段
vocab,embeddings = [],[]
with open('glove.6B.100d.txt','rt') as fi:
    full_content = fi.read().strip().split('\n')
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    vocab.append(i_word)
    embeddings.append(i_embeddings)
vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)
#insert '<pad>' and '<unk>' tokens at start of vocab_npa.
vocab_npa = np.insert(vocab_npa, 0, '<pad>')
vocab_npa = np.insert(vocab_npa, 1, '<unk>')
print(vocab_npa[:10])

pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

#insert embeddings for pad and unk tokens at top of embs_npa.
embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))

# 第二段
nltk.download('punkt')
Train_File = pd.read_csv('DLtrain.csv')
enc_sent = list()
for idx in range(Train_File.shape[0]):
    temp1 = Train_File['Description'][idx]
    temp1 = temp1.replace('\\', ' ')
    temp1 = temp1.replace('-', ' ')
    while True:
        if 'http' in temp1:
            I = temp1.find('&lt')
            II = temp1.find('A&gt')
            unwant = temp1[I:II + 4]
            temp1 = temp1.replace(unwant, ' ')
        else:
            break
    line = temp1
    line = line.lower()
    temp = word_tokenize(line)
    alist = [word for word in temp]
    enc_sent.append(alist)

# 最後一段
import torch
import torch.nn as nn
def data_setting(sentences1, en_dict):
    enc_input = list()
    N = len(sentences1)
    for idx in range(N):
        n = len(sentences1[idx])
        alist = list()
        for i in range(100):
            if i < n:
                word = sentences1[idx][i]
                if word not in en_dict:
                    alist.append(1)
                else:
                    alist.append(en_dict.index(word))
            else:
                alist.append(0)
        enc_input.append(alist)
        print("\r完成進度{0}".format((idx + 1)/N), end='')
    return torch.LongTensor(enc_input)
enc_input = data_setting(enc_sent, list(vocab_npa))