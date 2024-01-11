#! /usr/bin/env python
import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from model import Model
import random
import csv
import math
import pickle
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
#print("tensorflow version: " + str(tf.__version__))
D = "uniform"
batch_size = 50
#filename="monotonic"
filename = "train_adv_combine"
# filename = "train"
maxM = 100000
path = False
eps = [.15]
delta = [.3]
#eps = [.001,.01,.05,.1, .15]
#delta = [.001,.01,.05,.15,.3]

# Load HIDOST training dataset
test_data = '../data/traintest_all_500test/test_data.libsvm'
x_test, y_test = datasets.load_svmlight_file(test_data,
                                    n_features=3514,
                                    multilabel=False,
                                    zero_based=False,
                                    query_id=False)
x_test = x_test.toarray()
x_test = [[float(0)]*3514]

#initialize empty model object to load into
model = tf.keras.Model()
#configure hyperparams
checkpoint = tf.train.Checkpoint(model)
print("filename: " + str(filename))
#configure to read from correct model file if using one of the NNs
if filename=="train_adv_combine":
    PATH = "../models/adv_trained/baseline_adv_combine_two.ckpt"
if filename=="baseline":
    PATH = "../models/adv_trained/baseline_checkpoint.ckpt"
if filename!="monotonic":
    checkpoint.restore(PATH)

model = tf.keras.Model()
checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(PATH)
    
#if monotonic model then load from xgboost instead of tensorflow
xgb_model = None
if filename == "monotonic":
    xgb_model = xgb.Booster()
    xgb_model.load_model("monotonic_xgb.json")

#calculate data dimensions
n_obs = len(x_test)
n_features = len(x_test[0])

#eps = [.9]
#delta = [.9]

p = np.floor(np.log2(np.sqrt(n_features/np.log2(n_features))))

def mutate(x,y,k=1,path=False):
    #if using path-connected neighbor for mutations, 
    #find some tau-sized subset of 0 coordinates to increment
    if path:
        k = random.randint(1,p)
        tau = 2**k
        # print("p: "+ str(p)+", k: "+ str(k)+", tau: " + str(tau))
        zeroInds = np.where(x==0)[0]
        np.random.shuffle(zeroInds)
        zeroInds = zeroInds[:tau]
        x[zeroInds]=1
        return x
    #if edge test, simply find a hamming neighbor
    #only increment in the direction that could possibly cause non-monotonicity
    else:
        for i in range(k):
            inds = np.where(x==1-y)[0]
            newInd = random.choice(inds)
            x[newInd]=1-x[newInd]
        return x


def testBatch(x,xgb_mod=None,cap=None,centered=True,path=False):
    if not cap:
        cap=len(x_test)
    x = x[:cap]
    xNew = []
    x_mutated = []

    # for the within strategy, which selects existing neighbors 
    # from our valid set (usually test data)
    if not centered:
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
                    reachedCap = True
                    break
            # print("progress: " + str(float(count)/float(num_total)))
            if reachedCap:
                break
        if not reachedCap:
            print("ran out of examples using within-distribution strategy, setting result to N/A")
            return "N/A"
    # this code block for the uniform and centered-in strategy, which both use "mutations"
    else:
        xNew = x.copy()
        if xgb_mod:
            dtest = xgb.DMatrix(xNew)
            preds = xgb_model.predict(dtest)
            y = [1 if p > 0.5 else 0 for p in preds]
        else:
            y = model.predict(tf.convert_to_tensor(xNew))
            
        y=y[:cap]
        for i in range(len(xNew)):
            x_mutated.append(mutate(xNew[i],y[i],k=1,path=path))
        if xgb_mod: 
            dmutated = xgb.DMatrix(x_mutated)
            mutated_preds = xgb_mod.predict(dmutated)
            y_mutated = [1 if p > 0.5 else 0 for p in mutated_preds]
        else:
            y_mutated = model.predict(tf.convert_to_tensor(x_mutated))   
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
    

    for e in eps:
        for d in delta:
            np.random.shuffle(x_test)
            
            if path:
                m = int(np.ceil(np.log(1/d)/np.log(np.sqrt(n_features)/(np.sqrt(n_features)-(e**2)))))
            else:
                m=int(np.ceil(np.log(1/d)/np.log(n_features/(n_features-e))))

            print("delta: " + str(d)+ ", epsilon: " + str(e) + ", m: " + str(m))
            
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
                success = testBatch(x_test,cap=m,xgb_mod=xgb_model,centered=False,path =path)
            else:
                maxRounds = 0
                for r in range(numRounds):
                    #print("progress: " + str(float(r)/float(numRounds)))
                    if D=="centered_in":
                        x_input = x_test
                    elif D=="uniform":
                        x_input = [np.random.randint(2,size=n_features) for _ in range(len(x_test))]                           
                    if testBatch(x_input,xgb_mod = xgb_model,path=path) == "Reject":
                        print("encountered rejection")
                        success="Reject"
                        maxRounds = r
                        break
                if success == "Accept":
                    maxRounds = numRounds
                if D=="centered_in":
                    x_input = x_test
                elif D=="uniform":
                    x_input = [np.random.randint(2,size=n_features) for _ in range(len(x_test))]
                #print("rounds completed: " + str(maxRounds) + " out of " + str(numRounds))
                if testBatch(x_input,xgb_mod = xgb_model,cap = remainderRound,path=path ) == "Reject":
                    success="Reject"
                            
            writer.writerow([e, d,success])