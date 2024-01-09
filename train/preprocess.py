#! /usr/bin/env python
import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow.compat.v1 as tf
from model import Model
import random
import csv
import math
import pickle
tf.disable_eager_execution()
tf.disable_v2_behavior()
D = "uniform"
batch_size = 50
filename="monotonic"
maxM = 10000
path = False

print('Loading regular testing datasets...')
test_data = '../data/traintest_all_500test/test_data.libsvm'
x_test, y_test = datasets.load_svmlight_file(test_data,
                                    n_features=3514,
                                    multilabel=False,
                                    zero_based=False,
                                    query_id=False)
x_test = x_test.toarray()


model = Model()

model.tf_interval1(batch_size)

if filename=="train_adv_combine":
    PATH = "../models/adv_trained/baseline_adv_combine_two.ckpt"
if filename=="baseline":
    PATH = "../models/adv_trained/baseline_checkpoint.ckpt"
xgb_model = None
if filename == "monotonic":
    xgb_model = xgb.Booster()
    xgb_model.load_model("monotonic_xgb.json")
    
    

learning_rate = tf.placeholder(tf.float32)
saver = tf.train.Saver()
maxM = 100000

n_obs = len(x_test)
n_features = len(x_test[0])

#eps = [.9]
#delta = [.9]
eps = [.01,.05,.1, .15]
delta = [.01,.05,.15,.3]


def mutate(x,y,k=1,path=False):
    if path:
        p = 2**np.floor(np.log2(np.sqrt(n_features/np.log2(n_features))))
        k = random.randint(1,p)
        tau = 2**k
        print("tau: " + str(tau))
        zeroInds = np.where(x==0)[0]
        zeroInds.shuffle()
        zeroInds = zeroInds[:tau]
        x[zeroInds]=1
        return x
    else:
        for i in range(k):
            inds = np.where(x==1-y)[0]
            newInd = random.choice(inds)
            x[newInd]=1-x[newInd]
        return x
    
def testBatch(x,sess=None,xgb_mod=None,cap=None,centered=True,path=False):
    print("cap: " + str(cap) + ", len(x at top): " + str(len(x)))
    if not cap:
        cap=len(x_test)
    if not sess and not xgb_mod:
        print("NEED EITHER THE MONOTONIC MODEL OR NN")
        return   
    print("cap: " + str(cap))     
    if cap==1: 
        cap=2
    x = x[:cap]
    xNew = []
    x_mutated = []

    # centered is the hamming neighbor selection strategy
    # using mutation for centered-in and uniform distributions
    if centered:
        xNew = x.copy()
        y = sess.run(model.y_pred,\
                feed_dict={model.x_input:x,\
            })
        y=y[:cap]
        for i in range(len(xNew)):
            x_mutated.append(mutate(xNew[i],y[i],k=1,path=path))
        y_mutated = sess.run(model.y_pred,\
                    feed_dict={model.x_input:x_mutated.copy(),\
            })
    # for the within strategy, which selects existing neighbors 
    # from our valid set (usually test data)
    else:
        # print("y_p up top: " + str(y_p))
        num_total = math.comb(len(x),2)
        count=0
        reachedCap = False
        
        for i in range(len(x)):
            x1 = x[i]
            for j in range(i):
                count+=1
                x2 = x[j]
                #if using edge test, just search for comparable neighboring points
                if not path:
                    x1 = x1.astype(bool)
                    x2 = x2.astype(bool)
                    xorsum = np.sum(np.bitwise_xor(x1, x2))
                    if xorsum==1:
                        xNew.append(x1.astype(int))
                        x_mutated.append(x2.astype(int))
                # if using path test, just search for comparable points
                else:
                    diffVec = x1-x2
                    maxDiff = np.max(diffVec)
                    minDiff = np.min(diffVec)
                    if (maxDiff>0 and minDiff==0) or (maxDiff==0 and minDiff<0):
                        xNew.append(x1.astype(int))
                        x_mutated.append(x2.astype(int))
                        
                if len(x_mutated)==cap:
                    print("length of dataset: " + str(len(x_mutated)) + ", cap: " + str(cap))
                    reachedCap = True
                    break
            # print("progress: " + str(float(count)/float(num_total)))
            if reachedCap:
                break
        if not reachedCap:
            print("ran out of examples using within-distribution strategy, setting result to N/A")
            return "N/A"
        dtest = xgb.DMatrix(xNew.copy())
        preds = xgb_model.predict(dtest)
        y = [1 if p > 0.5 else 0 for p in preds]
        dmutated = xgb.DMatrix(x_mutated)
        mutated_preds = xgb_mod.predict(dmutated)
        y_mutated = [1 if p > 0.5 else 0 for p in mutated_preds]
        
        
    maxRows = 0
    for i in range(len(x_mutated)):
        sum_orig = str(np.sum(xNew[i]))
        sum_mutated = str(np.sum(x_mutated[i]))
        mutated_pred = str(y_mutated[i])
        orig_pred = str(y[i])
        if (sum_mutated>sum_orig and mutated_pred<orig_pred) or (sum_mutated<sum_orig and mutated_pred>orig_pred):
            print("HIT TEST FAILURE ON ROW " + str(i)+", sum_orig: " + str(sum_orig)+ ", sum_mutated: " + str(sum_mutated) + ", mutated_pred: " + str(mutated_pred) + ". orig_pred: " + str(orig_pred))
            return "Reject"
    return "Accept"

with open(filename+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epsilon", "delta","success"])
    with tf.Session() as sess:
        print("filename: " + str(filename))
        if filename!="monotonic":
            saver.restore(sess, PATH)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        if file=="train_adv_combine":
            PATH = "../models/adv_trained/baseline_adv_combine_two.ckpt"
        if file=="baseline":
            PATH = "../models/adv_trained/baseline_checkpoint.ckpt"

        for e in eps:
            for d in delta:
                np.random.shuffle(x_test)
                
                print("p: " + str(p))
                if path:
                    m = int(np.ceil(np.log(1/d)/np.log(n_features/(np.sqrt(n_features)-(e**2)))))
                else:
                    m=int(np.ceil(np.log(1/d)/np.log(n_features/(n_features-e))))

                print("delta: " + str(d)+ ", epsilon: " + str(e) + ", m: " + str(m))
                
                y_p=None 
                success="Accept"
                if m>maxM:
                    print("setting success to N/A")
                    success = "N/A"
                    writer.writerow([e, d,success])
                    continue
                
                numRounds = m//len(x_test)
                remainderRound = m%len(x_test)
                    # print("maxM: " + str(maxM) + ", len(x_test): " + str(len(x_test)) + ", numRounds: " + str(numRounds) + ", remainder: " + str(remainderRound)) 
                
                if D=="within":
                    success = testBatch(x_test,sess,cap=m,xgb_mod=xgb_model,centered=False,path =path)
                else:
                    maxRounds = 0
                    for r in range(numRounds):
                        print("progress: " + str(float(r)/float(numRounds)))
                        if D=="centered_in":
                            x_input = x_test
                        elif D=="uniform":
                            x_input = [np.random.randint(2,size=n_features) for _ in range(len(x_test))]
                            print('done initializing random matrix')
                        if testBatch(x_input,y_p,sess,xgb_mod = xgb_model,path=path) == "Reject":
                            success="Reject"
                            maxRounds = r
                            break
                    if D=="centered_in":
                        x_input = x_test
                    elif D=="uniform":
                        x_input = [np.random.randint(2,size=n_features) for _ in range(len(x_test))]
                    print("rounds completed: " + str(maxRounds) + " out of " + str(numRounds))
                    if testBatch(x_input,y_p,sess,xgb_mod = xgb_model,cap = remainderRound,path=path ) == "Reject":
                        success="Reject"
                                
                writer.writerow([e, d,success])