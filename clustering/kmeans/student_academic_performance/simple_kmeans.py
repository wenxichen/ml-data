#!/usr/bin/env python

'''
Created by Wenxi Chen.
Perform simple kmeans clustering for the student academic dataset.
Use kmeans++ to initialize the centroids.
The dataset is created by 
Elaf Abu Amrieh, Thair Hamtini, and Ibrahim Aljarah, The University of Jordan, Amman, Jordan.
The easy way to find the dataset is from IbrahimAljarah on Kaggle
at https://www.kaggle.com/aljarah/xAPI-Edu-Data.

To visualize the clustered data, 
I only pick out the "raised hand" and "Discussion" attrubutes.
Clearly, there's not much cluster from the look of the data.
So clustering on these two is not so meaningful beside demonstrating the implementation of kmeans.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def kmeanspp(X, K):
    ''' implement kmeans++ for initializing centroids. '''
    
    c = np.empty((K, X.shape[1]))
    X_copy = np.copy(X)
    
    # randomly select first centroid
    rint = np.random.randint(X.shape[0])
    c[0,:] = X[rint]
    X_copy = np.delete(X_copy, (rint), axis=0)

    # use minimum distance based distribution to select other centroids
    for i in range(1, K):
        n = X_copy.shape[0]
        dist = np.sum(np.square(X_copy - c[1,:]), 1).reshape(n,1)
        for j in range(1, i):
            temp_dist = np.sum(np.square(X_copy - c[j,:]), 1).reshape(n,1)
            dist = np.hstack((dist, temp_dist))
        dist = np.min(dist, 1)
        
        distribution = dist / float(np.sum(dist))
        c_idx = np.random.choice(n, p=distribution)
        c[i,:] = X_copy[c_idx]
        X_copy = np.delete(X_copy, (c_idx), axis=0) 
        
        
    return c

def Kmeans(X, K):
    ''' implement kmeans clustering '''
    
    
    n = X.shape[0]
    # c = X[np.random.choice(n, K, replace=False)]      # random initialize the centroids
    c = kmeanspp(X, K)      # K x d    initialize centroids using kmeans++
    # print "initial centroids", c
    c_temp = np.empty(c.shape)
    a = np.empty((n,1))
    
    converged = False
    while not converged:
        
        # assign each data point to the closest centroid
        for i in range(n):
            a[i] = np.argmin(np.sum(np.square(c - X[i]), 1))
            
        for k in range(K):
            c_temp[k] = np.mean(X[(a == k).ravel()], 0)
        if np.array_equal(c_temp, c):
            converged = True  
        c = c_temp
    
    return a, c

if __name__ == "__main__":
    
    # because there's no missing value or curruption,
    # no need to do data cleasing
    student_academic = pd.read_csv('xAPI-Edu-Data.csv')
    X = student_academic.loc[:,['raisedhands', 'Discussion']].as_matrix()
    K = 3   
    
    a, c = Kmeans(X, 3)
    
    a = a.ravel()
    plt.scatter(X[a==0][:,0], X[a==0][:,1], color="red")
    plt.scatter(X[a==1][:,0], X[a==1][:,1], color="green")
    plt.scatter(X[a==2][:,0], X[a==2][:,1], color="blue")
    plt.show()
    