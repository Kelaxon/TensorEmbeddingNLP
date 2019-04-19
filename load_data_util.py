import os
import nltk
import collections
from nltk.tokenize.toktok import ToktokTokenizer
import pandas as pd
import config

def Path2Sentence(file_path):
    file = open(file_path)
    sentences = file.read().split('\n')
    sentences = sentences[0:-1]

    return sentences


def BuildVocabulary(X):
    max_sentence_len = 0
    word_frequency = collections.Counter()
    for line in X:
        words = _WordTokenize(line.lower())

        if len(words) > max_sentence_len:
            max_sentence_len = len(words)

        for word in words:
            word_frequency[word] += 1
    return word_frequency, max_sentence_len


def Sentence2Index(X, word2index):
    """
    Return a list of each index sentences like [12, 23, 51] <-([I, am, happy])
    """

    Xout = []

    for line in X:
        words = _WordTokenize(line.lower())
        sequence = []

        for word in words:
            if word in word2index:
                sequence.append(word2index[word])
            else:
                sequence.append(word2index["<UNK>"])

        Xout.append(sequence)

    return Xout


def _WordTokenize(sentence):
    """
    For the different data sets, different tokenizing methods are needed.
    url_mode: the url will be held in output.
    e.g. This is https://www.google.com -> [This, is, https://www.google.com]
    """
    if config.CONFIG['TOKEN_MODE'] == 'DEFAULT':
        return nltk.word_tokenize(sentence)
    elif config.CONFIG['TOKEN_MODE'] == 'URL':
        return ToktokTokenizer().tokenize(sentence)
    else:
        raise NotImplementedError("_WordTokenize: mode %s is not implemented!" % CONFIG['TOKEN_MODE'])


def CleanTwitter(twitter):
    """
    Clean special characters in users' twitter like '\t\t', etc
    """
    # TODO considering other special cases is needed
    return twitter.replace("\t\t", "")


def BuildTruthTXT(ROOT_PATH):
    """
    Return a pandas Dataframe builded from truth.txt
    """

    return pd.read_csv(os.path.join(ROOT_PATH, 'truth.txt'), engine='python', sep=':::', names= \
        ['userid', 'gender', 'age_group', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open'])

