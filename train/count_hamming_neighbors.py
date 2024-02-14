import pickle
import numpy as np
import math
from sklearn import datasets

# test_data = '../data/traintest_all_500test/test_data.libsvm'
# x_test, y_test = datasets.load_svmlight_file(test_data,
#                                     n_features=3514,
#                                     multilabel=False,
#                                     zero_based=False,
#                                     query_id=False)
# x = x_test.toarray()

train_data = '../data/traintest_all_500test/train_data.libsvm'
x_train, y_train = datasets.load_svmlight_file(train_data,
                                    n_features=3514,
                                    multilabel=False,
                                    zero_based=False,
                                    query_id=False)
x = x_train.toarray()
print("x size: " + str(len(x)))
num_total = math.comb(len(x),2)
hamming_pairs=[]
unique_neighbors = []
count=0
print("x[0]: " + str(x[0]))
for i in range(len(x)):
    x1 = x[i]
    for j in range(i):
        count+=1
        x2 = x[j]
        diffVec = x1-x2
        maxDiff = np.max(diffVec)
        minDiff = np.min(diffVec)
        if ((maxDiff>0 and minDiff==0) or (maxDiff==0 and minDiff<0)):
            x1 = x1.astype(bool)
            x2 = x2.astype(bool)
            hamming_pairs.append((x1,x2))
    #print("progress: " + str(float(count)/float(num_total)))
print("percentage of unique points with neighbors in V: " + str(len(unique_neighbors)))
print("number of hamming pairs: " +str(len(hamming_pairs)))