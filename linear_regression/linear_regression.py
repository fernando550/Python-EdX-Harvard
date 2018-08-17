# linear regression

# quantitative = regression
# qualitative = classification
# loss function for regression = squared error loss
# loss function for classification = 0 - 1 function

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import statsmodels.api as sm

# simple linear regression
n = 100
beta0 = 5
beta1 = 2
np.random.seed(1)
x = 10 * ss.uniform.rvs(size=n)
y = beta0 + beta1 * x + ss.norm.rvs(loc=0,scale=1, size=n) # random noise

#plt.figure()
#plt.plot(x,y,"o", ms=5)
#xx = np.array([0,10])
#plt.plot(xx, beta0 + beta1 * xx)
#plt.xlabel("x")

# residual sum of squares (RSS)
#def compute_rss(y_estimate, y): 
#  return sum(np.power(y-y_estimate, 2)) 
#def estimate_y(x, b_0, b_1): 
#  return b_0 + b_1 * x 
#rss = compute_rss(estimate_y(x, beta0, beta1), y)

# residual sum of squares (RSS) with multiple slope values
#rss =[]
#slopes = np.arange(-10,15,0.01)
#for m in slopes:
#    rss.append(np.sum((y-beta0-m*x)**2))
#
#min_val = np.argmin(rss)
#print("min square error: ", min_val)
#
#plt.figure()
#plt.plot(slopes,rss)
#plt.xlabel("slope")
#plt.ylabel("rss")

# true estimation using statsmodels.api module
X = sm.add_constant(x) #add a y intercept to the x array
mod = sm.OLS(y, X) # ordinary least squares
est = mod.fit()
print(est.summary())

# sampling distributions of parameter estimations
# R^2 = TSS-RSS / TSS







