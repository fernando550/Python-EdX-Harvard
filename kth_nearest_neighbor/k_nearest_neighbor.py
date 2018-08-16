# K-Nearest Neighbors

import numpy as np
import random
import scipy.stats as ss
import matplotlib.pyplot as plt

def distance(p1,p2):
    """Find distance between two points"""
    return np.sqrt(np.sum(np.power(p2-p1,2)))

def majority_vote(votes):
    """Find mode of an array, picks a tie at random"""
    vote_counts ={}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] +=1
        else:
            vote_counts[vote] = 1

    winners = []
    max_count = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)

    return random.choice(winners)
    
def kNN(p, points, k=5):
    """Find the k nearest neighbors of point p and return indices"""
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p,points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    """Predict class based on kNN"""
    ind = kNN(p, points, k)
    return majority_vote(outcomes[ind])

# distance between each point in an array and a single point
# points = np.floor(5*np.random.random((9,2)))
# p = np.floor(5*np.random.random((1,2)))
# k = np.random.randint(3,10)
# outcomes = np.array([0,0,0,0,1,1,1,1,1])
#
# knn_pred = knn_predict(p, points, outcomes, k)
# print(knn_pred)

def generate_data(n=50):
    """generate two sets of points from bivariate normal dist"""
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))), axis=0)
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1,n)))
    return (points, outcomes, n)

# n=20
# (points, outcomes) = generate_data(n)
#
# plt.figure()
# plt.plot(points[:n,0],points[:n,1], "ro")
# plt.plot(points[n:,0],points[n:,1], "bo")

def make_pred_grid(predictors, outcomes, limits, h, k):
    (xmin, xmax, ymin, ymax) = limits
    xs = np.arange(xmin, xmax, h)
    ys = np.arange(ymin,ymax, h)
    xx, yy = np.meshgrid(xs, ys)
    
    prediction_grid = np.zeros(xx.shape, dtype=int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid [i,j] = knn_predict(p,predictors, outcomes, k)
            
    return (xx, yy, prediction_grid)

def plot_prediction_grid (xx, yy, prediction_grid):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    
(predictors, outcomes, n) = generate_data()

k=5; limits = (-3,4,-3,4); h = 0.1
(xx,yy,prediction_grid) = make_pred_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx,yy,prediction_grid)



###############################################################################
###############################################################################
###############################################################################

# kNN using SkiKitLearn

from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
predictors = iris.data[:,0:2]
outcomes = iris.target

plt.plot(predictors[outcomes==0][:,0],predictors[outcomes==0][:,1], "ro")
plt.plot(predictors[outcomes==1][:,0],predictors[outcomes==1][:,1], "go")
plt.plot(predictors[outcomes==2][:,0],predictors[outcomes==2][:,1], "bo")

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(predictors, outcomes)
predictions = knn.predict(predictors)