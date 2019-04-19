import pandas
import os
import numpy as np

from load_data_util import *

def LoadSST1(ROOT_PATH):
    corpus = pandas.read_pickle(os.path.join(ROOT_PATH, 'SST1.pkl'))
    X, y = list(corpus.sentence), list(corpus.label)

    word_frequency, max_sentence_len = BuildVocabulary(X)

    max_vocab = len(word_frequency)
    word2index = {x[0]: i + 2 for i, x in enumerate(word_frequency.most_common(max_vocab))}
    word2index["<PAD>"] = 0
    word2index["<UNK>"] = 1
    index2word = {v: k for k, v in word2index.items()}

    X = Sentence2Index(X, word2index)

    vocabulary = [index2word[i] for i in range(max_vocab + 2)]

    Xlist = []
    for x in X:
        if len(x) <= max_sentence_len:
            Xlist.append(list(x) + [0] * (max_sentence_len - len(x)))
        else:
            raise ValueError

    X = np.array(Xlist)
    y = np.array(y)
    vocabulary = np.array(vocabulary)

    return X, y, vocabulary


def LoadSST2(ROOT_PATH):
    corpus = pandas.read_pickle(os.path.join(ROOT_PATH, 'SST2.pkl'))
    X, y = list(corpus.sentence), list(corpus.label)

    word_frequency, max_sentence_len = BuildVocabulary(X)

    max_vocab = len(word_frequency)
    word2index = {x[0]: i + 2 for i, x in enumerate(word_frequency.most_common(max_vocab))}
    word2index["<PAD>"] = 0
    word2index["<UNK>"] = 1
    index2word = {v: k for k, v in word2index.items()}

    X = Sentence2Index(X, word2index)

    vocabulary = [index2word[i] for i in range(max_vocab + 2)]

    Xlist = []
    for x in X:
        if len(x) <= max_sentence_len:
            Xlist.append(list(x) + [0] * (max_sentence_len - len(x)))
        else:
            raise ValueError

    X = np.array(Xlist)
    y = np.array(y)
    vocabulary = np.array(vocabulary)

    return X, y, vocabulary

def LoadCR(ROOT_PATH):
    corpus = pandas.read_pickle(os.path.join(ROOT_PATH, 'CR.pkl'))
    X, y = list(corpus.sentence), list(corpus.label)

    word_frequency, max_sentence_len = BuildVocabulary(X)

    max_vocab = len(word_frequency)
    word2index = {x[0]: i + 2 for i, x in enumerate(word_frequency.most_common(max_vocab))}
    word2index["<PAD>"] = 0
    word2index["<UNK>"] = 1
    index2word = {v: k for k, v in word2index.items()}

    X = Sentence2Index(X, word2index)

    vocabulary = [index2word[i] for i in range(max_vocab + 2)]

    Xlist = []
    for x in X:
        if len(x) <= max_sentence_len:
            Xlist.append(list(x) + [0] * (max_sentence_len - len(x)))
        else:
            raise ValueError

    X = np.array(Xlist)
    y = np.array(y)
    vocabulary = np.array(vocabulary)

    return X, y, vocabulary

def LoadSUBJ(ROOT_PATH):
    corpus = pandas.read_pickle(os.path.join(ROOT_PATH, 'SUBJ.pkl'))
    X, y = list(corpus.sentence), list(corpus.label)

    word_frequency, max_sentence_len = BuildVocabulary(X)

    max_vocab = len(word_frequency)
    word2index = {x[0]: i + 2 for i, x in enumerate(word_frequency.most_common(max_vocab))}
    word2index["<PAD>"] = 0
    word2index["<UNK>"] = 1
    index2word = {v: k for k, v in word2index.items()}

    X = Sentence2Index(X, word2index)

    vocabulary = [index2word[i] for i in range(max_vocab + 2)]

    Xlist = []
    for x in X:
        if len(x) <= max_sentence_len:
            Xlist.append(list(x) + [0] * (max_sentence_len - len(x)))
        else:
            raise ValueError

    X = np.array(Xlist)
    y = np.array(y)
    vocabulary = np.array(vocabulary)

    return X, y, vocabulary