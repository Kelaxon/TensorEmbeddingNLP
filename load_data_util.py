import nltk
import collections

def Path2Sentence(file_path):
    file = open(file_path)
    sentences = file.read().split('\n')
    sentences = sentences[0:-1]

    return sentences

def BuildVocabulary(X):
    max_sentence_len = 0
    word_frequency = collections.Counter()
    for line in X:
        words = nltk.word_tokenize(line.lower())

        if len(words) > max_sentence_len:
            max_sentence_len = len(words)

        for word in words:
            word_frequency[word] += 1
    return word_frequency, max_sentence_len


def Sentence2Index(X, word2index):
    Xout = []

    for line in X:
        words = nltk.word_tokenize(line.lower())
        sequence = []

        for word in words:
            if word in word2index:
                sequence.append(word2index[word])
            else:
                sequence.append(word2index["<UNK>"])

        Xout.append(sequence)

    return Xout
