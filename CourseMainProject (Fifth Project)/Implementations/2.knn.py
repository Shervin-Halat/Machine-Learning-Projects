import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

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
y = np.array(data[:,19],dtype = int)

###########KNN:
scores = []
for n in range(15):
    #kn = KNeighborsClassifier(n_neighbors=n).fit(x,y)
    score = cross_val_score(KNeighborsClassifier(n_neighbors=n+1),x,y,cv = 10)
    scores.append(score.mean())

index = scores.index(max(scores))
y_pred = KNeighborsClassifier(n_neighbors= index+1).fit(x,y).predict(x)

print('accuracy of KNN:') 
print(scores[index+2]*100)
print('confusion matrix of KNN:') 
print(confusion_matrix(y, y_pred))
