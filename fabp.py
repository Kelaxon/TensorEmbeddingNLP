import argparse
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matlab.engine
import scipy.io
import os

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

nbrs = NearestNeighbors(n_neighbors=args.neighbor, algorithm='kd_tree', metric='euclidean').fit(doc2vec)
distances, indices = nbrs.kneighbors(doc2vec)
index_i = []
index_j = []
val = []
for i in range(indices.shape[0]):
    for item in list(indices[i])[1:]:
        index_i.append(indices[i][0])
        index_j.append(item)
        val.append(1)

label_train = list(y_train)
for i, l in enumerate(label_train):
    if l==0:
        label_train[i]=2
label_test = [0]*y_test.shape[0]
labels = np.array(label_train+label_test)

scipy.io.savemat('tmp_matrix_info.mat', dict(index_i=index_i, index_j=index_j,\
                                             val=val, label=labels))

eng = matlab.engine.start_matlab()
eng.FaBP(nargout=0)
eng.quit()
pred = scipy.io.loadmat('tmp_label_mat.mat')

os.remove("tmp_matrix_info.mat")
os.remove("tmp_label_mat.mat")

corr_number = 0
for ind in inds_test:
    if pred['final_labels'][ind]==y[ind]:
        corr_number += 1

print('Accuracy:%f'%(corr_number/len(inds_test)))