import numpy as np
import joblib
import os
import argparse

from config import *
from load_data_util import *
from load_data_semeval2007 import SemEval2007_Path2String

from sklearn import preprocessing
import xml.etree.ElementTree as ET


def _Load16000Oneliners(ROOT_PATH):
    file_path_pos = os.path.join(ROOT_PATH, 'Jokes16000-utf8.txt')
    file_path_neg = os.path.join(ROOT_PATH, 'MIX16000-utf8.txt')

    X1 = Path2Sentence(file_path_pos)
    X2 = Path2Sentence(file_path_neg)
    y1 = [1]*len(X1)
    y2 = [0]*len(X2)

    X = X1 + X2
    y = y1 + y2

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

def _LoadSemEval2007(ROOT_PATH):
    DATA_X_PATH1 = os.path.join(ROOT_PATH, 'AffectiveText.test/affectivetext_test.xml')
    DATA_Y_PATH1 = os.path.join(ROOT_PATH, 'AffectiveText.test/affectivetext_test.emotions.gold')

    DATA_X_PATH2 = os.path.join(ROOT_PATH, 'AffectiveText.trial/affectivetext_trial.xml')
    DATA_Y_PATH2 = os.path.join(ROOT_PATH, 'AffectiveText.trial/affectivetext_trial.emotions.gold')

    X1, y1 = SemEval2007_Path2String(DATA_X_PATH1, DATA_Y_PATH1)
    X2, y2 = SemEval2007_Path2String(DATA_X_PATH2, DATA_Y_PATH2)
    X = X1 + X2
    y = y1 + y2

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

    ylist = []
    for val in y:
        val_sum = float(sum(val))
        if val_sum == 0:
            ylist.append([1 / 6.0] * 6)
        else:
            ylist.append([val / val_sum for val in val])

    X = np.array(Xlist)
    y = np.array(ylist)
    vocabulary = np.array(vocabulary)

    return X, y, vocabulary

def _LoadAuthorProfiling(ROOT_PATH):
    """
    Load the authoer profiling dataset from files,
    and then return a index matrix and a list type of the vocabulary.
    Attention: this dataset does not contain any label (y).
    """
    global DEBUG_NUM
    label_name = 'stable'

    # Prepare
    content_fnames = [s for s in os.listdir(ROOT_PATH) if s.endswith('.xml')]
    user_ids = [id.split('.')[0] for id in content_fnames]
    truth_df = BuildTruthTXT(ROOT_PATH).set_index('userid')
    ordered_truth_df = truth_df.loc[user_ids]  # ordered_truth_df should be in a same order with content_fnames

    # Label encoding e.g. [0.2, 0.3, 0.1] -> [0, 1, 2]
    le = preprocessing.LabelEncoder()
    le.fit(ordered_truth_df[label_name])
    users_labels = le.transform(ordered_truth_df[label_name])
    print("Feature '{}' of authors has {} classes. \n{}".format(label_name, len(le.classes_), str(le.classes_)))

    # build X
    X = []
    y = []
    # for uid, content_fname in enumerate(content_fnames[:DEBUG_NUM]):
    for uid, content_fname in enumerate(content_fnames):
        root = ET.parse(os.path.join(ROOT_PATH, content_fname)).getroot()  # parse XML
        sentence_list = [CleanTwitter(child.text) for child in root]  # build a sentence list of each user's data
        X += sentence_list  # combine all the users
        # build label for each user
        y += [users_labels[uid]] * len(sentence_list)

    # build vocabulary
    word_frequency, max_sentence_len = BuildVocabulary(X)

    max_vocab = len(word_frequency)
    word2index = {x[0]: i + 2 for i, x in enumerate(word_frequency.most_common(max_vocab))}
    word2index["<PAD>"] = 0
    word2index["<UNK>"] = 1
    index2word = {v: k for k, v in word2index.items()}

    X = Sentence2Index(X, word2index)

    vocabulary = [index2word[i] for i in range(max_vocab + 2)]

    # sentence alignment with filling zero, e.g. [87, 31, 52, 0, 0, 0]
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


def LoadData(config):
    # Calling a function of a module by using its name (a string)
    print("Data is rebuilding...")
    _LoadDataMethod = globals()[config['LOAD_DATA_METHOD']]
    return _LoadDataMethod(config['ROOT_PATH'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", help="use which dataset", default='16000oneliners')
    parser.add_argument('--label_portion', help='proportion of labels', type=float, default=0.5)
    parser.add_argument("--data_portion", help="proportion of data",
                        type=float, default=0.3)  # Enhance debug efficiency
    parser.add_argument("--seed", help="reproducible experiment with seeds", type=int, default=666)
    parser.add_argument("--rebuild", help="ignore the existing data and rebuild ones",
                        dest='rebuild', action='store_true')
    parser.add_argument("--no-rebuild", help="load existing data directly", dest='rebuild', action='store_false')
    parser.set_defaults(rebuild=False)

    args = parser.parse_args()
    CONFIG = GetConfig(args.option)

    if not args.rebuild:
        [X, y, vocabulary] = joblib.load(CONFIG['SAVED_RAW_DATA'])
    else:
        X, y, vocabulary = LoadData(CONFIG)
        joblib.dump([X, y, vocabulary], CONFIG['SAVED_RAW_DATA'])

    N = len(X)
    random_generator = np.random.RandomState(args.seed)
    inds_all = random_generator.permutation(N)

    X = X[inds_all]
    y = y[inds_all]

    inds_all = np.array(range(N))
    test_num = int((1 - args.label_portion) * N)
    inds_train = inds_all[:N - test_num]
    inds_test = inds_all[N - test_num:]

    X_train = X[inds_train]
    y_train = y[inds_train]
    X_test = X[inds_test]
    y_test = y[inds_test]

    if not os.path.exists(os.path.join(CONFIG['ROOT_PATH'], 'data')):
        os.makedirs(os.path.join(CONFIG['ROOT_PATH'], 'data'))
    joblib.dump([vocabulary, X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all], \
                os.path.join(CONFIG['ROOT_PATH'], 'data/raw.pkl'))

    print("Data loading finished.\n"+\
        "train data number: {}, test data number: {}.".format(len(X_train), len(X_test)))
