#! /usr/bin/env python
import numpy as np
from sklearn import datasets
import tensorflow.compat.v1 as tf
from model import Model
import random
import csv
import math
import argparse
from tqdm import tqdm
import xgboost as xgb

tf.disable_eager_execution()
tf.disable_v2_behavior()
strat = tf.distribute.MirroredStrategy()
print("tensorflow version: " + str(tf.__version__))
gpus = tf.config.list_physical_devices('GPU')
print("using GPU? " + str(gpus))

def parse_args():
    parser = argparse.ArgumentParser(description='Regular training and robust training of the pdf malware classification model.')
    parser.add_argument('--model_name', type=str, help='Load checkpoint from \{monotonic|robust_monotonic|robust_combine_three|train_adv_combine\} model.',required=True)
    parser.add_argument('-D', type=str, help='Use \{uniform|centered|empirical\} strategy for pair selection.', required=True)    
    parser.add_argument('--edge', action='store_true', default=False)
    parser.add_argument('--maxM', type=int, default=10000000)
    parser.add_argument('--maxM', type=int, default=10000000)
    parser.add_argument('--train_only',action='store_true',default=False)


    return parser.parse_args()

args = parse_args()
path = not args.edge
D = args.D
maxM = args.maxM
filename = args.model_name
train_only = args.train_only

batch_size = 50
maxM = 10000000
#eps = [.4]
#delta = [.4]
eps = [.01,.1,.5,.9]
delta = [.01,.1,.5,.9]

train_data = '../data/traintest_all_500test/train_data.libsvm'
x_train, y_train = datasets.load_svmlight_file(train_data,
                                    n_features=3514,
                                    multilabel=False,
                                    zero_based=False,
                                    query_id=False)
x_train = x_train.toarray()

print("x_train.shape: " + str(x_train.shape))

# Load HIDOST training dataset
test_data = '../data/traintest_all_500test/test_data.libsvm'
x_test, y_test = datasets.load_svmlight_file(test_data,
                                    n_features=3514,
                                    multilabel=False,
                                    zero_based=False,
                                    query_id=False)
x_test = x_test.toarray()

if train_only:
    x_test,y_test = x_train,y_train

#initialize empty model object to load into
model = Model()
model.tf_interval1(batch_size)
#configure hyperparams
learning_rate = tf.placeholder(tf.float32)
saver = tf.train.Saver()
with strat.scope():
    with tf.Session() as sess:
        print("filename: " + str(filename))
        #configure to read from correct model file if using one of the NNs
        if filename=="train_adv_combine":
            PATH = "../models/adv_trained/baseline_adv_combine_two.ckpt"
        if filename=="baseline":
            PATH = "../models/adv_trained/test_model_name.ckpt"
        if filename=="robust_monotonic":
            PATH="../models/adv_trained/robust_monotonic.ckpt"
        if filename=="robust_combine_three":
            PATH="../models/adv_trained/robust_combine_three.ckpt"
        if filename=="robust_combine_two":
            PATH="../models/adv_trained/robust_combine_two.ckpt"

        if filename!="monotonic":
            saver.restore(sess, PATH)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())        

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
        tau_max = 2**p
        print("tau_max: " + str(tau_max))
        
        def mutate(x,y,k=1,path=False):
            #if using path-connected neighbor for mutations, 
            #find some tau-sized subset of 0 coordinates to increment
            xCopy=x.copy()

            if path:
                k = random.randint(1,p)
                tau = 2**k
                #print("p: "+ str(p)+", k: "+ str(k)+", tau: " + str(tau))
                zeroInds = np.where(x==0)[0]
                np.random.shuffle(zeroInds)
                zeroInds = zeroInds[:tau]
                #print("zeroInds: " + str(zeroInds))
                xCopy[zeroInds]=1
                #print("sum x:" + str(sum(x)) + ", sum xCopy: " + str(sum(xCopy)))
            #if edge test, simply find a hamming neighbor
            #only increment in the direction that could possibly cause non-monotonicity
            else:
                for i in range(k):
                    inds = np.where(x==1-y)[0]
                    newInd = random.choice(inds)
                    xCopy[newInd]=1-xCopy[newInd]
            return xCopy
            

        def testBatch(x,xgb_mod=None,cap=None,centered=True,path=False):
            
            if not cap:
                cap=len(x_test)
            if not sess and not xgb_mod:
                print("NEED EITHER THE MONOTONIC MODEL OR NN")
                return   
            xNew = []
            x_mutated = []
            y_mutated = []
            # for the empirical strategy, which selects existing neighbors 
            # from our valid set (usually test data)
            if not centered:
                # print("y_p up top: " + str(y_p))
                num_total = math.comb(len(x),2)
                count=0
                reachedCap = False
                print("cap: " + str(cap) + ", len(x): " + str(len(x)))
                if (cap>22000):
                    print("ran out of examples using empirical-distribution strategy, setting result to N/A")
                    return "N/A"

                for i in tqdm(range(len(x)), desc="Finding Pairs within distribution"):
                    x1 = x[i]
                    for j in range(i):
                        count+=1
                        x2 = x[j]
                        #if using edge test, search for comparable neighboring points

                        if not path:
                            x1 = x1.astype(bool)
                            x2 = x2.astype(bool)
                            xorsum = np.sum(np.bitwise_xor(x1, x2))
                            if xorsum==1:
                                xNew.append(x1.astype(int))
                                x_mutated.append(x2.astype(int))
                        # if using path test, search for comparable points (maybe need to restrict to distance tau?)
                        else:
                            diffVec = x1-x2
                            maxDiff = np.max(diffVec)
                            minDiff = np.min(diffVec)
                            if ((maxDiff>0 and minDiff==0) or (maxDiff==0 and minDiff<0)):
                                x1 = x1.astype(bool)
                                x2 = x2.astype(bool)
                                xorsum = np.sum(np.bitwise_xor(x1, x2))
                                if xorsum<tau_max:
                                    xNew.append(x1.astype(int))
                                    x_mutated.append(x2.astype(int))
                                
                        if len(x_mutated)==cap:
                            reachedCap = True
                            break
                    # print("progress: " + str(float(count)/float(num_total)))
                    if reachedCap:
                        break
                print("len xNew: " + str(len(xNew)))
                if xgb_mod:
                    dtest = xgb.DMatrix(xNew)
                    preds = xgb_model.predict(dtest)
                    y = [1 if p > 0.5 else 0 for p in preds]
                    dmutated = xgb.DMatrix(x_mutated)
                    mutated_preds = xgb_mod.predict(dmutated)
                    y_mutated = [1 if p > 0.5 else 0 for p in mutated_preds]
                else:
                    y = sess.run(model.y_pred,\
                            feed_dict={model.x_input:xNew
                    })
                    y_mutated = sess.run(model.y_pred,\
                            feed_dict={model.x_input:x_mutated.copy()
                    })
            
            # this code block for the uniform and centered-in strategy, which both use "mutations"
            else:
                xNew = x.copy()
                xNew = xNew[:cap]

                if xgb_mod:
                    dtest = xgb.DMatrix(xNew)
                    preds = xgb_model.predict(dtest)
                    y = [1 if p > 0.5 else 0 for p in preds]
                else:
                    y = sess.run(model.y_pred,\
                            feed_dict={model.x_input:xNew
                    })
                y=y[:cap]
                for i in range(len(xNew)):
                    x_mutated.append(mutate(xNew[i],y[i],k=1,path=path))
                    #print("sum xNew: " + str(sum(xNew[i])) + ", sum x_mutated: " + str(sum(x_mutated[i])))
                if xgb_mod: 
                    dmutated = xgb.DMatrix(x_mutated)
                    mutated_preds = xgb_mod.predict(dmutated)
                    y_mutated = [1 if p > 0.5 else 0 for p in mutated_preds]
                else:
                    y_mutated = sess.run(model.y_pred,\
                            feed_dict={model.x_input:x_mutated.copy()
                    })
            maxRows = 0
            for i in range(len(x_mutated)):
                #print("sum xNew: " + str(sum(xNew[i])) + ", sum x_mutated: " + str(sum(x_mutated[i])))
                sum_orig = str(np.sum(xNew[i]))
                sum_mutated = str(np.sum(x_mutated[i]))
                mutated_pred = str(y_mutated[i])
                orig_pred = str(y[i])
                #print("sum_orig: " + str(sum_orig)+ ", sum_mutated: " + str(sum_mutated) + ", mutated_pred: " + str(mutated_pred) + ". orig_pred: " + str(orig_pred))

                if (sum_mutated>sum_orig and mutated_pred<orig_pred) or (sum_mutated<sum_orig and mutated_pred>orig_pred):
                    print("HIT TEST FAILURE ON ROW " + str(i)+", sum_orig: " + str(sum_orig)+ ", sum_mutated: " + str(sum_mutated) + ", mutated_pred: " + str(mutated_pred) + ". orig_pred: " + str(orig_pred))
                    return "Reject"
            return "Accept"
        pathString = "" if path else "_edge"
        with open("tests/"+filename+"_"+D+pathString+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epsilon", "delta","success"])
            

            for e in eps:
                for d in delta:
                    np.random.shuffle(x_test)
                    small_m = int(np.ceil(np.log(1/d)/np.log(np.sqrt(n_features)/(np.sqrt(n_features)-(e**2)))))
                    big_m=int(np.ceil(np.log(1/d)/np.log(n_features/(n_features-e))))
                    print("small_m: " + str(small_m) + ", big_m: " + str(big_m))
                    if path:
                        print("goin with small m")
                        m=small_m
                    else:
                        print("going with big m")
                        m=big_m


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
                    
                    if D=="empirical":
                        success = testBatch(x_test,cap=m,xgb_mod=xgb_model,centered=False,path =path)
                    else:
                        maxRounds = 0
                        for r in tqdm(range(numRounds),desc = "Generating mutations and evaluating in batches of size " + str(len(x_test))):
                            print("progress: " + str(float(r)/float(numRounds)))
                            if D=="centered":
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
                        if D=="centered":
                            x_input = x_test
                        elif D=="uniform":
                            x_input = [np.random.randint(2,size=n_features) for _ in range(len(x_test))]
                        print("rounds completed: " + str(maxRounds) + " out of " + str(numRounds))
                        if testBatch(x_input,xgb_mod = xgb_model,cap = remainderRound,path=path ) == "Reject":
                            success="Reject"
                                    
                    writer.writerow([e, d,success])