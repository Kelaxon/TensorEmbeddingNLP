from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import argparse
import joblib
import scipy.io
import numpy as np

from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--option", help="use which dataset", default='semeval2007')
parser.add_argument("--neighbor", help="number of neighbors", type=int, default=10)
args = parser.parse_args()

CONFIG = GetConfig(args.option)

[vocabulary, X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] = \
    joblib.load(CONFIG['RAW_DATA'])
doc2vec = joblib.load(CONFIG['TENSOR_EMBEDDING'])

# propagation

n_samples, n_classes = y.shape
labels = np.zeros((n_samples, n_classes))

for i, y in enumerate(y_train):
    labels[i] = y

step = y_train.shape[0]
for i, y in enumerate(y_test):
    labels[i+step] = -1

label_prop_model = LabelPropagation(kernel='knn', n_neighbors=args.neighbor)
label_prop_model.fit(doc2vec, labels)

pred = label_prop_model.predict_proba(doc2vec)

scipy.io.savemat(CONFIG['RESULT'], dict(pred=pred[inds_test], true=y_test))
# print('Accuracy:%f'%label_prop_model.score(doc2vec[inds_test],y_test))