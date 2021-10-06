import numpy as np
from os import listdir
import matplotlib.pyplot as plt

directory_path = '.'
file_types = ['npy', 'npz']

np_vars = {dir_content: np.load(dir_content)
           for dir_content in listdir(directory_path)
           if dir_content.split('.')[-1] in file_types}
           
vars = list(np_vars.values())
data = np.stack(vars)

print(np.shape(data))

#plt.scatter(data[:,0], np.log(data[:,1]))


X = (data[:,0])
y = np.log(data[:,1])

X = np.power(data[:,0], 1/10)
#y = data[:,1]
y = (data[:,1])

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

print(np.shape(X))
print(np.shape(y))

import statsmodels.api as sm

mod = sm.OLS(X,y)

fii = mod.fit()
print(fii.summary2())

from sklearn.metrics import r2_score

print(r2_score(y, X))

plt.scatter(y, X)
#plt.ylim([0, 5])
#plt.xlim([0, 15])
#plt.plot(t,T)
plt.show()




#from sklearn.linear_model import LinearRegression
#reg = LinearRegression().fit(X, y)
#print(reg.summary())
#print(reg.score(X, y))
#print(reg.coef_)
#print(reg.intercept_)

#plt.subplot(212)
#plt.scatter(np.log(data[:,0]), np.power(data[:,1],2))
#plt.show()


