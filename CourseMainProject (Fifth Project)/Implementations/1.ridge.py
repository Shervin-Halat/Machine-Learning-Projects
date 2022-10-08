import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import Ridge

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

loc = 'C:/Users/sherw/OneDrive/Desktop/ML_Project/Dataset/Regression/day.csv'
data = np.array(list(csv.reader(open(loc))))[1:,2:]
x = np.array(data[:,:-1],dtype = float)
y = np.array(data[:,-1],dtype = float)
np.random.shuffle(data)

###########ridge regression:
score = []
f = list(range(5,14))
scores = []
parameters = {'alpha' : [0.1,0.2,0.5,0.8,1,2,5]}
for i in f:
    x_new = SelectKBest(chi2, k=i).fit_transform(x, y)
    ridge = Ridge()
    ridge_regressor = GridSearchCV(ridge,parameters,scoring = 'neg_mean_squared_error',cv = 10)
    ridge_regressor.fit(x_new,y)
    scores.append(ridge_regressor.best_score_)

maxscores = min(scores)
print('ridge regression score: {}'.format(maxscores * -1))
