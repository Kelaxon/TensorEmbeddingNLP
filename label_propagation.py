from sklearn.semi_supervised import LabelPropagation
import argparse
import joblib
import numpy as np

from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--option", help="use which dataset", default='16000oneliners')
parser.add_argument("--neighbor", help="number of neighbors", type=int, default=10)
args = parser.parse_args()

CONFIG = GetConfig(args.option)

[vocabulary, X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] = \
    joblib.load(CONFIG['RAW_DATA'])
doc2vec = joblib.load(CONFIG['TENSOR_EMBEDDING'])

# propagation

label_train = list(y_train)
label_test = [-1]*y_test.shape[0]
labels = np.array(label_train+label_test)

label_prop_model = LabelPropagation(kernel='knn', n_neighbors=args.neighbor)
label_prop_model.fit(doc2vec, labels)

# pred = label_prop_model.predict(doc2vec)

print('Accuracy:%f'%label_prop_model.score(doc2vec[inds_test],y_test))