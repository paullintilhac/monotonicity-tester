import pickle
import numpy as np
import csv

# file = "train_adv_combine"
file = "monotonic"
# file = "train"


with open(file+'.pickle', 'rb') as f:
    x = pickle.load(f)

eps = list(np.power(float(10),range(-4,0)))
# eps.append(float(.02))
# eps.append(float(.03))
# eps.append(float(.04))
# eps.append(float(.05))
n_obs = len(x)
print("n_obs: " + str(n_obs))
n_features = len(x[0]["x_test"])
print("n_features: " + str(n_features))

print("eps: "+ str(eps))
delta = [.01,.03,.05,.07,.09,.11]
print("delta: " +str(delta))

with open(file+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epsilon", "delta","success"])

    for e in eps:
        for d in delta:
            np.random.shuffle(x)
            m=int(np.ceil(np.log(1/d)/np.log(n_features/(n_features-e))))
            print("m: " + str(m))

            success="Accept"
    
            if m>len(x):
                print("setting success to N/A")
                success = "N/A"
            else:
                for i in range(m):
                    # print("i: " + str(i))
                    # print(x[i].keys())
                    if D=="centered_in":
                        mutated_pred = x[i]["mutated_pred"]
                        orig_pred = x[i]["orig_pred"]
                        sum_mutated = x[i]['sum_mutated']
                        sum_orig = x[i]['sum_orig']
                    elif D=="uniform":
                        orig_pred = np.random.randint(2, size=n_features)

                    # print("sum_mutated: " + str(sum_mutated) + ", sum_orig: " + str(sum_orig))
                    # print("mutated_pred: " + str(mutated_pred) + ", orig_pred: " + str(orig_pred))

                    if sum_mutated>sum_orig and mutated_pred<orig_pred:
                        success="Reject"
                        break
                    if sum_mutated<sum_orig and mutated_pred>orig_pred:
                        success="Reject"
                        break
            print("success after loop: " + str(success))
            writer.writerow([e, d,success])

        
        
