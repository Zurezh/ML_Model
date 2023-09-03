# -*- coding: utf-8 -*-
"""
Pizza Price Predictor

@author: Suresh Manem
"""

import numpy as np
import matplotlib.pyplot as plt

# X represents the features of our training data, the diameters of the pizzas. 
# A scikit-learn convention is to name the matrix of feature vectors X.  
# Uppercase letters indicate matrices, and lowercase letters indicate vectors. 
X = np.array([[6], [8], [10], [14], [18]]).reshape(-1, 1) 
y = [7, 9, 13, 17.5, 18]  # y is a vector representing the prices of the pizzas.

plt.figure() 
plt.title('Pizza price plotted against diameter') 
plt.xlabel('Diameter in inches') 
plt.ylabel('Price in dollars') 
plt.plot(X, y, 'k.') 
plt.axis([0, 25, 0, 25]) 
plt.grid(True) 
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

test_pizza = np.array([[12]])
predicted_price = model.predict(test_pizza)[0]
print('A 12" pizza should cost : $%.2f' % predicted_price)

print('Residual Sum of Squares (RSS) is %.2f' % np.mean((model.predict(X)-y)**2))

x_bar = X.mean()
print('Mean of diameter ',x_bar)

# Note that we subtract one from the number of training instances when calculating the sample variance. 
# This technique is called Bessel's correction. It corrects the bias in the estimation of the population variance
# from a sample.
variance = ((X - x_bar)**2).sum() / (X.shape[0] - 1)
print('The variance is ',variance)
print('Finding variance using numpy ',np.var(X,ddof=1))

# We previously used a List to represent y.
# Here we switch to a NumPy ndarray, which provides a method to calulcate the sample mean.
y = np.array([7, 9, 13, 17.5, 18])

y_bar = y.mean()
# We transpose X because both operands must be row vectors
covariance = np.multiply((X - x_bar).transpose(), y - y_bar).sum() / (X.shape[0] - 1)
print('The covariance is ',covariance)
print('Finding covariance using numpy ',np.cov(X.transpose(), y)[0][1])


print('Access whether the above model is good?') 
X_train = np.array([6, 8, 10, 14, 18]).reshape(-1, 1) 
y_train = [7, 9, 13, 17.5, 18] 
 
X_test = np.array([8, 9, 11, 16, 12]).reshape(-1, 1) 
y_test = [11, 8.5, 15, 18, 11]  
 
model.fit(X_train, y_train) 
r_squared = model.score(X_test, y_test) 
print('The r_squared must be postive between 0 and 1: ',r_squared )