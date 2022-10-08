import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import randint
from math import sqrt
'''
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
'''

loc = 'C:/Users/sherw/OneDrive/Desktop/HW4_ml/dataset/data_kmeans_4.txt'
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
DB = []
temp = []
#for iteration in range(n):
label_data = [[],[]]

######
def s1s2(l1 , l2):
    s1 = sqrt(np.mean(np.linalg.norm((l1 - np.mean(l1,axis = 0)),axis = 1)**2))
    s2 = sqrt(np.mean(np.linalg.norm((l2 - np.mean(l2,axis= 0)),axis = 1)**2))
    return (s1 + s2)

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
    DB.append(s1s2(label_data[0],label_data[1]) / np.linalg.norm(center[0]-center[1]))
    j.append(sse)
    label_data = [[],[]]

print(counter)     
for i in range(2):
    label_data[i]= np.array(label_data[i])

plt.scatter(label_data[0][:,0],label_data[0][:,1],s = 2, label = '1st Cluster')
plt.scatter(label_data[1][:,0],label_data[1][:,1],s = 2, label = '2nd Cluster')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means Clustering of data_kmeans_4')
plt.legend()
plt.show()


plt.plot(list(range(len(DB))), DB)
plt.scatter(list(range(len(DB))), DB , c= 'red')
plt.title('Davies-Bouldin index of each iteration of data_kmeans_4')
plt.xlabel('#iteration')
plt.ylabel('DB')
plt.xticks(list(range(len(DB))))
plt.show()

