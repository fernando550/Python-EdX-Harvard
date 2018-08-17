# multi linear regression using scikit

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# scikit for multi linear regression
n=500
b0 = 5
b1 = 2
b2 = -1
np.random.seed(1)
x1 = 10 * ss.uniform.rvs(size=n)
x2 = 10 * ss.uniform.rvs(size=n)
y = b0 + b1 * x1 + b2 * x2 + ss.norm.rvs(loc=0, scale=1, size=n)

X = np.stack([x1,x2], axis=1) # aggregate both x arrays

lm = LinearRegression(fit_intercept=True)
#lm.fit(X, y)
#print("model generated: ", lm.intercept_, lm.coef_[0], lm.coef_[1])

#X0 = np.array([2,4])
#print("model predictions with: ", lm.predict(X0.reshape(1,-1)))

#print("R^2 value: ", lm.score(X,y))

# train test split for training a model
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.5, random_state=1)
lm.fit(xtrain,ytrain)
print("accuracy: ", lm.score(xtest,ytest))






