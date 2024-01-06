#! /usr/bin/env python
import os
import time
import argparse
import numpy as np
import pickle
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow.compat.v1 as tf
from model import Model
import scipy
import xgboost as xgb
import random
import csv
tf.disable_eager_execution()
tf.disable_v2_behavior()
D = "centered_in"
batch_size = 50
file="train_adv_combine"


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
print("HELLO")
if file=="train_adv_combine":
    PATH = "../models/adv_trained/baseline_adv_combine_two.ckpt"
if file=="baseline":
    PATH = "../models/adv_trained/baseline_checkpoint.ckpt"
print("initial path: " + PATH)
xgb_model = None
if file == "monotonic":
    xgb_model = xgb.Booster()
    xgb_model.load_model("monotonic_xgb.json")

learning_rate = tf.placeholder(tf.float32)
saver = tf.train.Saver()
maxM = 10000000
# maxM = 12000

eps = [.01,.05,.15,.25,.4]
n_obs = len(x_test)
print("x_test[0]: " +str(len(x_test[0])))
n_features = len(x_test[0])

delta = [.01,.05,.08,.013,.19,.25]


def mutate(x,y,k=1):
    inds = np.where(x==1-y)[0]
    for i in range(k):
        newInd = random.choice(inds)
        x[newInd]=1-x[newInd]
    return x

def testBatchCentered(x,sess=None,xgb_mod=None,cap=None):

    if not cap:
        cap=len(x_test)

    if not sess and not xgb_mod:
        print("NEED EITHER THE MONOTONIC MODEL OR NN")
        return
    
     # print("cap: " + str(cap))
    x = x[:cap]
    arr=[]
    x_mutated = []
    for i in range(len(xNew)):
        x_mutated.append(mutate(xNew[i],y_p[i],k=1))

    xNew = x.copy()

    if xgb_mod: 
        
        dtest = xgb.DMatrix(xNew, label=y_test)
        dmutated = xgb.DMatrix(x_mutated,label=y_test)

        preds = xgb_mod.predict(dtest)
        print(preds[2372])
        y_p = [1 if p > 0.5 else 0 for p in preds]
        
        mutated_preds = xgb_mod.predict(dmutated)
        y_mutated = [1 if p > 0.5 else 0 for p in mutated_preds]
        
    #if NN model
    else:
        y_p = sess.run(model.y_pred,\
                                    feed_dict={model.x_input:x_test.copy(),\
                                    model.y_input:y_test.copy()
                                    })
        
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

with open(file+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epsilon", "delta","success"])

    with tf.Session() as sess:
        if (file!="monotonic"):
            saver.restore(sess, PATH)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        else: sess=None
        for e in eps:
            for d in delta:
                np.random.shuffle(x_test)
                m=int(np.ceil(np.log(1/d)/np.log(n_features/(n_features-e))))
                print("delta: " + str(d)+ ", epsilon: " + str(e) + ", m: " + str(m))   
                
                if D=="centered_in":
                    
                    numRounds = maxM//len(x_test)
                    remainderRound = maxM%len(x_test)
                    # print("maxM: " + str(maxM) + ", len(x_test): " + str(len(x_test)) + ", numRounds: " + str(numRounds) + ", remainder: " + str(remainderRound)) 
                    success=True
                    maxRounds = 0
                    for r in range(numRounds):
                        print("progress: " + str(float(r)/float(numRounds)))
                        if not testBatchCentered(x_test,sess,xgb_mod=xgb_model):
                            success=False
                            maxRounds = r
                            break
                    print("rounds completed: " + str(maxRounds) + " out of " + str(numRounds))
                    if not testBatchCentered(x_test,sess,xgb_mod = xgb_model,cap = remainderRound ):
                        success=False
                    
                if D=="uniform":
                    x_orig = np.random.randint(2, size=(n_obs,n_features))

                    
                writer.writerow([e, d,success])