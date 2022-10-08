import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import randint
'''
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
'''

loc = 'C:/Users/sherw/OneDrive/Desktop/HW4_ml/dataset/data_kmeans_1.txt'
file = open(loc)
data = np.array(list(csv.reader(file)),dtype = float)
print(data)
print(data.shape)
np.random.shuffle(data)
'''
plt.scatter(data[:,0],data[:,1], s = 0.2)
plt.show()
'''

### k-means ###

k = 2
center = []
for n in range(2):
    center.append([])
    center[n] = data[randint(0,len(data)-1)]

#n = 4000
counter = 0
j = []
temp = []
#for iteration in range(n):
label_data = [[],[]]

while True:
    counter += 1
    sse = 0
    for i in data:
        #print(i)
        #print(type(i))
        dist0 = np.linalg.norm(i - center[0])
        dist1 = np.linalg.norm(i - center[1])
        if dist0 <= dist1:
            label_data[0].append(list(i))
        else:
            label_data[1].append(list(i))
    if temp == label_data:
        break
    temp = label_data
    for c in range(2):
            center[c] = np.mean(label_data[c],axis = 0)
            for i in label_data[c]:
                sse += np.linalg.norm(i - center[c]) ** 2
    j.append(sse)
    label_data = [[],[]]

print(counter)     
for i in range(2):
    label_data[i]= np.array(label_data[i])

plt.scatter(label_data[0][:,0],label_data[0][:,1],s = 2, label = '1st Cluster')
plt.scatter(label_data[1][:,0],label_data[1][:,1],s = 2, label = '2nd Cluster')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means Clustering of data_kmeans_1')
plt.legend()
plt.show()

print(j)
plt.plot(list(range(len(j))), j)
plt.scatter(list(range(len(j))), j , c= 'red')
plt.title('Jsse of each iteration of data_kmeans_1')
plt.xlabel('#iteration')
plt.ylabel('Jsse')
plt.xticks(list(range(len(j))))
plt.show()

