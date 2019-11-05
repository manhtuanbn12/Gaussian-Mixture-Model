# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:57:36 2019

@author: Tuan
"""
from collections import defaultdict
import numpy as np

# import variable
with open('20news-train-processed.txt') as f:
    corpus = [(line.split("<fff>")[0], line.split("<fff>")[2]) for line in f.read().splitlines()]
    train_labels = [i[0] for i in corpus]
    train_data = [i[1] for i in corpus]

doc_count = defaultdict(int)
corpus_size = len(train_labels)
for data in train_data:
    text = data.split(" ")
    words = list(set(text))
    for word in words:
        doc_count[word] += 1
        
vocab = []
train_processed = []
for data in train_data:
    text = data.split(" ")
    text_processed = [word for word in text if (doc_count[word] > 10 and not word.isdigit())]
    train_processed.append(" ".join(text_processed))
    for word in text_processed:
        vocab.append(word)
    
vocab = list(set(vocab))
vocab_dict = {vocab[i]:i for i in range(len(vocab))}

# tf-idf features
from sklearn.feature_extraction.text import CountVectorizer
tfidf_vect = CountVectorizer()
X_train_tfidf = tfidf_vect.fit_transform(train_processed)

N = X_train_tfidf.shape[0]
M = X_train_tfidf.shape[1]
N1 = 100

# Gaussian Mixture Model
# Khoi tao pi, nuy, sigma
K = 20
pi = np.zeros([K])
pi[0] = 1
nuy = np.zeros([N1,M])
sigma = np.eye(M)
X_train = X_train_tfidf.toarray()
X_train = X_train * 1./np.sum(np.sum(X_train))

for i in range(50):
    #update T
    T = np.zeros([N1, K])
    mau = 0
    for k in range(K):
        for n in range(N1):
            mau += pi[k]*np.exp(-1./2 * np.sum((X_train[n,:]-nuy[k,:])**2))

    for n in range(N1):
        for k in range(K):
            T[n,k] = pi[k]*np.exp(-1./2 * np.sum((X_train[n,:]-nuy[k,:])**2)) / mau

    # Update pi
    for k in range(K):
        pi[k] = np.sum(T[:,k])/N1

    # Update nuy
    nuy = np.zeros([K,M])
    for k in range(K):
        for n in range(N1):
            nuy[k,:] += T[n,k] * X_train[k,:]
    nuy = nuy * 1./np.sum(np.sum(T))