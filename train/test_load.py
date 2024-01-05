import pickle 

with open('train_adv_combine.pickle', 'rb') as handle:
    b = pickle.load(handle)

print((b[0]))