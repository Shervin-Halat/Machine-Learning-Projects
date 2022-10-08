import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

loc = 'C:/Users/sherw/OneDrive/Desktop/ML_Project/Dataset/Regression/day.csv'
data = np.array(list(csv.reader(open(loc))))[1:,2:]
x = np.array(data[:,:-1],dtype = float)
y = np.array(data[:,-1],dtype = float)
np.random.shuffle(data)

###########linear regression:
score = []
f = list(range(5,14))
scores = []

for i in f:
    x_new = SelectKBest(chi2, k=i).fit_transform(x, y)
    lin_regressor = LinearRegression()
    mse = cross_val_score(lin_regressor,x,y,scoring = 'neg_mean_squared_error',cv = 10)
    mean_mse = np.mean(mse)
    scores.append(mean_mse)

maxscore = max(scores)
print('linear regression score: {}'.format(maxscore))
