import pickle
import numpy as np
import math
from sklearn import datasets

test_data = '../data/traintest_all_500test/test_data.libsvm'
x_test, y_test = datasets.load_svmlight_file(test_data,
                                    n_features=3514,
                                    multilabel=False,
                                    zero_based=False,
                                    query_id=False)
x = x_test.toarray()

print("len(x): " + str(len(x)))

num_total = math.comb(len(x),2)
hamming_pairs=[]
unique_neighbors = []
count=0
print("x[0]: " + str(x[0]))
for i in range(len(x)):
    for j in range(i):
        count+=1
        x1 = x[i].astype(bool)
        x2 = x[j].astype(bool)
        xorsum = np.sum(np.bitwise_xor(x1, x2))
        if xorsum==1:
            if not any(np.array_equal(x1, c) for c in unique_neighbors):
                unique_neighbors.append(x1)
            if not any(np.array_equal(x2, c) for c in unique_neighbors):
                unique_neighbors.append(x2)
            hamming_pairs.append((x1,x2))
    #print("progress: " + str(float(count)/float(num_total)))
print("percentage of unique points with neighbors in V: " + str(len(unique_neighbors)))
print("number of hamming pairs: " +str(len(hamming_pairs)))