from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import argparse
import joblib
# import scipy.io
import numpy as np
from sklearn.metrics import accuracy_score

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

# propagation

classes = np.unique(y)

n_samples = y.shape[0]
n_classes = classes.shape[0]

labels = np.zeros((n_samples, n_classes))

for i, val in enumerate(y_train):
    labels[i][int(val)] = 1.0

step = y_train.shape[0]
for i, val in enumerate(y_test):
    labels[i+step] = -1

label_prop_model = LabelPropagation(kernel='knn', n_neighbors=args.neighbor)
label_prop_model.fit(doc2vec, labels)

pred_probability = label_prop_model.predict_proba(doc2vec)

pred_class = classes[np.argmax(pred_probability, axis=1)].ravel()

# scipy.io.savemat(CONFIG['RESULT'], dict(pred=pred_class[inds_test], true=y_test))
# scipy.io.savemat(CONFIG['RESULT'], dict(pred=pred[inds_train], true=y_train))
# scipy.io.savemat(CONFIG['RESULT'], dict(pred=pred[inds_test], true=y_test))


print('Accuracy:%f'%accuracy_score(y_test, pred_class[inds_test]))