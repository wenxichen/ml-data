import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Perfomr simple kmeans clustering for the student academic dataset
# the dataset is created by 
# Elaf Abu Amrieh, Thair Hamtini, and Ibrahim Aljarah, The University of Jordan, Amman, Jordan
# The easy way to find the dataset is from IbrahimAljarah on Kaggle
# at https://www.kaggle.com/aljarah/xAPI-Edu-Data

# To visualize the clustered data, 
# I only pick out the "raised hand" and "Discussion" attrubute
# Clearly, there's not much cluster from the look of the data
# So clustering on this two is not so meaningful 
# beside demonstrating the implementation of kmeans

def Kmeans(X, K):
    ''' implement kmeans clustering '''
    
    # c = kmeanspp(X, K)      # K x d
    
    n = X.shape[0]
    c = X[np.random.choice(n, K, replace=False)]
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
    