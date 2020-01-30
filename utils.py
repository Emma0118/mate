import numpy as np
import sys
import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib



def load_rotated_MNIST(datapath='MNIST_6rot.pkl.gz', left_out_idx=0):

    """
    6 domains, each input is (X, y)
    each domain contains 10 classes, and each class has 100 images.
    """
    file = gzip.open(datapath, 'rb')
    domains = pickle.load(file, encoding='latin1') # Python3 must use `encoding=latin1`
    # domains = pickle.load(gzip.open(datapath,'rb'))
    src_domains = domains[:]

    # print(type(domains[0]))

    del src_domains[left_out_idx]

    (X_test, y_test) = domains[left_out_idx]

    # print(len(src_domains)) # 5
    # print(X_test.shape) # (1000, 256)
    # print(y_test.shape)  # (1000,)

    # print(src_domains[0][1].shape)

    return src_domains, (X_test, y_test)



def construct_pair(X_list): # X_list: src_domain
    n_dom = len(X_list)
    X_in = np.vstack(X_list)
    X_outs = []
    for i in range(0, n_dom):
        X = X_list[i]

        Z_list = []
        for j in range(0, n_dom):
            Z_list.append(X)
        
        Z = np.vstack(Z_list)
     

        X_outs.append(Z)

        
    
    return X_in, X_outs

X_list = np.random.randn(3, 2, 1)
X_in, X_outs = construct_pair(X_list)

print(X_in.shape, len(X_outs))

print('X_in = ', X_in)

print('X_outs[0] = ', X_outs[0])

