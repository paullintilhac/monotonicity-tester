#! /usr/bin/env python
import os
import time
import argparse
import numpy as np
import pickle
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow.compat.v1 as tf
from model import Model
import scipy
import random
import csv
tf.disable_eager_execution()
tf.disable_v2_behavior()
D = "uniform"
batch_size = 50
filename="monotonic"


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
maxM = 10000000
# maxM = 12000

eps = [.01,.05,.15,.25,.4]
n_obs = len(x_test)
n_features = len(x_test[0])

delta = [.01,.05,.08,.013,.19,.25]


def mutate(x,y,k=1):
    inds = np.where(x==1-y)[0]
    for i in range(k):
        newInd = random.choice(inds)
        x[newInd]=1-x[newInd]
    return x
    
def testBatch(x,y,sess=None,xgb_mod=None,cap=None,centered=True):
    if not cap:
        cap=len(x_test)
    # print("cap: " + str(cap))
    
    if not sess and not xgb_mod:
        print("NEED EITHER THE MONOTONIC MODEL OR NN")
        return
    
    x = x[:cap]
    y=y[:cap]
    xNew = []
    x_mutated = []

    if centered:
        xNew = x.copy()
        for i in range(len(xNew)):
            x_mutated.append(mutate(xNew[i],y[i],k=1))
    else:
        for i in range(len(x)):
            for j in range(i):
                x1 = x[i]["x_test"].astype(bool)
                x2 = x[j]["x_test"].astype(bool)
                xorsum = np.sum(np.bitwise_xor(x1, x2))
                if xorsum==1:
                    xNew.append(x1)
                    x_mutated.append(x2)

    if xgb_mod: 
        dmutated = xgb.DMatrix(x_mutated)
        mutated_preds = xgb_mod.predict(dmutated)
        y_mutated = [1 if p > 0.5 else 0 for p in mutated_preds]
    else:
        y_mutated = sess.run(model.y_pred,\
                    feed_dict={model.x_input:x_mutated.copy(),\
                    model.y_input:y_test.copy()
                    })    
    # print("len(x): " + str(len(x)) + ", len(x_mutated): " + str(len(x_mutated)))
    maxRows = 0
    for i in range(len(x_mutated)):
        sum_orig = str(np.sum(x[i]))
        sum_mutated = str(np.sum(x_mutated[i]))
        mutated_pred = str(y_mutated[i])
        orig_pred = str(y_p[i])
        if sum_mutated>sum_orig and mutated_pred<orig_pred:
            print("HIT TEST FAILURE ON ROW " + str(i))
            return False
        if sum_mutated<sum_orig and mutated_pred>orig_pred:
            print("HIT TEST FAILURE ON ROW " + str(i))
            return False
    return True

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
                m=int(np.ceil(np.log(1/d)/np.log(n_features/(n_features-e))))
                print("delta: " + str(d)+ ", epsilon: " + str(e) + ", m: " + str(m))
                y_p=None 
                if (m>maxM):
                    m=maxM
                if xgb_model: 
                    dtest = xgb.DMatrix(x_test.copy())
                    preds = xgb_model.predict(dtest)
                    y_p = [1 if p > 0.5 else 0 for p in preds]
                else:
                    y_p = sess.run(model.y_pred,\
                                                feed_dict={model.x_input:x_test.copy(),\
                                                model.y_input:y_test.copy()
                                                })
                numRounds = m//len(x_test)
                remainderRound = m%len(x_test)
                    # print("maxM: " + str(maxM) + ", len(x_test): " + str(len(x_test)) + ", numRounds: " + str(numRounds) + ", remainder: " + str(remainderRound)) 
                success=True
                if D=="within":
                    if not testBatch(x_test,y_p,sess,xgb_mod=xgb_model,centered=False):
                        success=False
                else:
                    maxRounds = 0
                    for r in range(numRounds):
                        print("progress: " + str(float(r)/float(numRounds)))
                        if D=="centered_in":
                            x_input = x_test
                        elif D=="uniform":
                            x_input = [np.random.randint(2,size=n_features) for _ in range(len(x_test))]
                            print('done initializing random matrix')
                        if not testBatch(x_input,y_p,sess,xgb_mod = xgb_model):
                            success=False
                            maxRounds = r
                            break
                    if D=="centered_in":
                        x_input = x_test
                    elif D=="uniform":
                        x_input = [np.random.randint(2,size=n_features) for _ in range(len(x_test))]
                    print("rounds completed: " + str(maxRounds) + " out of " + str(numRounds))
                    if not testBatch(x_input,y_p,sess,xgb_mod = xgb_model,cap = remainderRound ):
                        success=False
                
                    
                writer.writerow([e, d,success])
                if r == numRounds-1:
                    maxRounds = r
                print("rounds completed: " + str(maxRounds) + " out of " + str(numRounds))   
                writer.writerow([e, d,success])