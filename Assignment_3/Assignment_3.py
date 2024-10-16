import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('linear_regression_data.csv', header=None)

data.columns = ['X','Y']

X = data['X'].values
Y = data['Y'].values

meanX = np.mean(X)
meanY = np.mean(Y)

covarianceXY = np.sum((X - meanX) * (Y - meanY))
varianceX = np.sum((X - meanX)**2)

slope = covarianceXY / varianceX
intercept = meanY - slope * meanX

Yprediction = slope * X + intercept

plt.scatter(X,Y, color='blue',label='Data points')
plt.plot(X,Yprediction, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression using Covariance Approach')
plt.show()
