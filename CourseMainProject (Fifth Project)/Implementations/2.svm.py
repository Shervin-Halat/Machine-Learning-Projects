import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

loc = 'C:/Users/sherw/OneDrive/Desktop/ML_Project/Dataset/Classification/hepatitis.data'
data = []
with open(loc) as f:
    reader = csv.reader(f)
    for line in reader:
        data.append(line)


data = np.array(data)
np.random.shuffle(data)

for i in range(data.shape[1]):
    summ = 0
    count = 0
    for j in range(len(data[:,i])):
        if data[j,i] != '?':
            summ += float(data[j,i])
            count += 1
    miu = summ / count
    data[:,i][data[:,i] == '?'] = miu

data = np.array(data,dtype = float)
    
x = data[:,:19]
x2 = np.array(x)
y = np.array(data[:,19],dtype = int)

########### feature selection:


########### svc:
scores = []

for i in range(11,20):
    x_new = SelectKBest(chi2, k=i).fit_transform(x, y)
    scores.append(cross_val_score(SVC(kernel='linear'),x_new,y,cv = 10).mean())

index = scores.index(max(scores))
print(index + 11)
x_new = SelectKBest(chi2, k=index + 11).fit_transform(x, y)
y_pred = SVC(kernel='linear').fit(x_new,y).predict(x_new)

print('accuracy of SVM:') 
print(scores[index]*100)
print('confusion matrix of SVM:') 
print(confusion_matrix(y, y_pred))




