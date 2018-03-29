# -*- coding: utf-8 -*-
# @Time     : 2017/11/12  下午8:13
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : svm.py
# @Software : PyCharm

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import sys

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Gaussian distribution
mu1 = np.array([[0, 6]])
Sigma1 = np.array([[2, 1], [1, 2]])
R1 = cholesky(Sigma1)
s1 = np.dot(np.random.randn(1100, 2), R1) + mu1
print(s1.shape)
y1 = np.zeros([s1.shape[0]])
print(y1.shape)


mu2 = np.array([[6.5, 7]])
Sigma2 = np.array([[2, 1], [1, 2]])
R2 = cholesky(Sigma2)
s2 = np.dot(np.random.randn(1100, 2), R2) + mu2
y2 = np.ones([s2.shape[0]])
print(s2.shape)
print(y2.shape)

mu3 = np.array([[2, 2]])
Sigma3 = np.array([[2, 1], [1, 2]])
R3 = cholesky(Sigma3)
s3 = np.dot(np.random.randn(200, 2), R3) + mu3
y3 = np.zeros([s3.shape[0]]) + 2
print(s3.shape)
print(y3.shape)


# plt.plot(s1[:,0],s1[:,1],'+')
# plt.plot(s2[:,0],s2[:,1],'*', color='r')
# plt.plot(s3[:,0],s3[:,1],'.', color='y')
# plt.show()

# import some data to play with
# Take the first two features. We could avoid this by using a two-dim dataset
X = np.concatenate([s1[:1000], s2[:1000], s3[:100]], axis=0)
y = np.concatenate([y1[:1000], y2[:1000], y3[:100]], axis=0)
print(X.shape)
print(y.shape)

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
model1 = (svm.SVC(kernel='linear', C=C))
print(model1)
models1 = model1.fit(X, y)

# title for the plots
titles = ('SVC with linear kernel--1k+1k+100')

# Set-up 2x2 grid for plotting.

plt.figure(1)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
plt.plot(s1[:,0],s1[:,1],'o', color='blue')
plt.plot(s2[:,0],s2[:,1],'^', color='orange')
plt.plot(s3[:,0],s3[:,1],'s', color='red')

ax = plt.axes()

plot_contours(ax, models1, xx, yy,
              cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors=None)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(titles)


mu4 = np.array([[2, 2]])
Sigma4 = np.array([[2, 1], [1, 2]])
R4 = cholesky(Sigma4)
s4 = np.dot(np.random.randn(900, 2), R4) + mu4
y4 = np.zeros([s4.shape[0]]) + 2
print(s4.shape)
print(y4.shape)


X = np.concatenate([X, s4], axis=0)
y = np.concatenate([y, y4], axis=0)
print(X.shape)
print(y.shape)


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
model2 = (svm.SVC(kernel='linear', C=C))
print(model2)
models2 = model2.fit(X, y)

# title for the plots
titles = ('SVC with linear kernel--1k+1k+1k')

# Set-up 2x2 grid for plotting.

plt.figure(2)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
plt.plot(s1[:,0],s1[:,1],'o', color='blue')
plt.plot(s2[:,0],s2[:,1],'^', color='orange')
plt.plot(s3[:,0],s3[:,1],'s', color='red')
plt.plot(s4[:,0],s4[:,1],'s', color='red')

ax = plt.axes()

plot_contours(ax, models2, xx, yy,
              cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors=None)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(titles)
# plt.show()
# sys.exit()

testX = np.concatenate([s1[1000:], s2[1000:], s3[100:]], axis=0)
testY = np.concatenate([y1[1000:], y2[1000:], y3[100:]], axis=0)
# plt.plot(testX[:,0],testX[:,1],'x', color='black')

pred1 = model1.predict(testX)
print(pred1)
print(testY)
print(pred1 - testY)
right1 = np.argwhere((pred1-testY)==0).shape[0]
print(right1)

pred2 = model2.predict(testX)
print(pred2)
print(testY)
print(pred2 - testY)
right2 = np.argwhere((pred2-testY)==0).shape[0]
print(right2)
plt.show()