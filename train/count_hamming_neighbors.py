import pickle
import numpy as np
import math
# file = "train_adv_combine"
file = "monotonic"
# file = "train"

with open(file+'.pickle', 'rb') as f:
    x = pickle.load(f)

num_total = math.comb(len(x),2)
hamming_pairs=[]
count=0
for i in range(len(x)):
    for j in range(i):
        count+=1
        x1 = x[i]["x_test"].astype(bool)
        x2 = x[j]["x_test"].astype(bool)
        xorsum = np.sum(np.bitwise_xor(x1, x2))
        if xorsum==1:
            hamming_pairs.append((x1,x2))
    print("progress: " + str(float(count)/float(num_total)))
print("number of hamming pairs: " +str(len(hamming_pairs)))