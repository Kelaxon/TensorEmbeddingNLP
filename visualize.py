import argparse
import joblib
from sklearn.neighbors import NearestNeighbors


from config import *
from load_data_util import *

parser = argparse.ArgumentParser()
parser.add_argument("--option", help="use which dataset", default='semeval2007')
parser.add_argument("--neighbor", help="number of neighbors", type=int, default=10)
args = parser.parse_args()

CONFIG = GetConfig(args.option)

[vocabulary, X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] = \
    joblib.load(CONFIG['RAW_DATA'])
doc2vec = joblib.load(CONFIG['TENSOR_EMBEDDING'])

nbrs = NearestNeighbors(n_neighbors=args.neighbor + 1, algorithm='kd_tree', \
                        metric='euclidean').fit(doc2vec)

def _FindKNNInds(idx):
    vec = doc2vec[idx]
    _, indices = nbrs.kneighbors(vec[None, :], n_neighbors=args.neighbor+1)
    return indices[0][1:]

def pause():
    programPause = input("Press the <ENTER> key to continue...")

for i, x in enumerate(X):
    print('------------source sentence------------')
    words = Idx2Word(x, vocabulary)
    print(' '.join(words))
    print(y[i])

    print('------------neighbor sentence------------')
    indices = _FindKNNInds(i)
    for idx in indices:
        words = Idx2Word(X[idx], vocabulary)
        print(' '.join(words))
        print(y[idx])

    pause()