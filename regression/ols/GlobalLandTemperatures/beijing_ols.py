import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def ols(tr_feat, tr_label):
    ''' perform ordinary least square on given on dimensional data. '''
    n = len(tr_feat)
    X = tr_feat
    Y = tr_label
    X = np.hstack((np.ones((n,1)),X))       # n x (d+1)
    inverse = np.linalg.inv(X.T.dot(X))
    beta = inverse.dot(X.T.dot(Y))          # (d+1) x 1
    
    error = np.sum(np.square(Y - X.dot(beta))) / float(n)
    
    return beta, error

if __name__ == "__main__":

    global_temp_city = pd.read_csv('GlobalLandTemperaturesByMajorCity.csv', parse_dates=[0])
    beijing_temp = global_temp_city[global_temp_city["City"] == "Peking"]
    
    # get clean data for beijing's avg temperture each august
    num_years = beijing_temp.shape[0] / 12
    beijing_avg_temp = np.array([]).reshape((0,2))
    for i in range(num_years):
        aug_temp = beijing_temp.iloc[i*12]
        if not math.isnan(aug_temp["AverageTemperature"]):
            beijing_avg_temp = np.vstack([beijing_avg_temp, np.array([aug_temp["dt"].year, aug_temp["AverageTemperature"]])])
    
    n = len(beijing_avg_temp)
    tr_feats = beijing_avg_temp[:,0].reshape(n,1)
    tr_labels = beijing_avg_temp[:,1].reshape(n,1)
    beta, error = ols(tr_feats, tr_labels)
    
    # plot the data
    plt.scatter(tr_feats, tr_labels)
    
    # plot the ols euquation
    x = np.linspace(1820,2012,2012-1820+1)
    y = beta[0] + beta[1]*x
    plt.plot(x,y)
    
    plt.title("Average Aug Temperature in Beijing")
    plt.show()     