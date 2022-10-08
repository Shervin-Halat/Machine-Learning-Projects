import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import sample
from random import randint
from math import sqrt

loc1 = 'C:/Users/sherw/OneDrive/Desktop/HW4_ml/dataset/data_kmeans_1.txt'
loc2 = 'C:/Users/sherw/OneDrive/Desktop/HW4_ml/dataset/data_h.xlsx'

data = pandas.read_excel(loc2,'Sheet1')
data = np.array([ data[column].values for column in data.columns]).T
np.random.shuffle(data)

### k-means ###

def sse(cluster):
    error = 0
    c = np.mean(cluster,axis = 0)
    for i in cluster:
        error += np.linalg.norm(i - c) ** 2
    return error



def kmeans(dataset , k):

    #center = sample(list(dataset), k)
    label_data = []
    temp = []
    center = []
    
    for n in range(k):
        label_data.append([])
        center.append([])
        center[n] = data[randint(0,len(data)-1)]

    while True:
        for i in dataset:
            dist = []
            for c in center:
                dist.append(np.linalg.norm(i - c))
            label_data[dist.index(min(dist))].append(list(i))

        if temp == label_data:
            break
            
        temp = list(label_data)
        
        for i in range(k):
            center[i] = np.mean(label_data[i],axis = 0)
            label_data[i] = []
        #label_data = [[]]*k

    for i in range(len(label_data)):
        label_data[i] = np.array(label_data[i])

    return label_data
'''
d = kmeans(data,5)
for i in d:
    plt.scatter(i[:,0],i[:,1],s = 1)
plt.show()
'''

def hier_dev(dataset,n):
    why = 0
    clusters = kmeans(dataset,4)
    while True:
        sses = []
        for cl in clusters:
            sses.append(sse(cl))
        ind = sses.index(max(sses))
        to_cluster = clusters[ind]
        del clusters[ind]
        new_two = kmeans(to_cluster,2)
        clusters += new_two
        for i in clusters:
            if i.size == 0:
                del clusters[clusters.index(i)]
                why += 1
        print(why)
        ########
        temp = list(clusters)
        for i in range(len(temp)):
            temp[i] = np.array(temp[i])
        for i in temp:
            plt.scatter(i[:,0],i[:,1],s = 2, label = 'Cluster {}'.format(n))
        plt.show()

        ########
        
        if len(clusters) == n:
            break
    for i in range(len(clusters)):
        clusters[i] = np.array(clusters[i])
    return clusters

c = hier_dev(data,8)

n = 0
for i in c:
    n += 1
    plt.scatter(i[:,0],i[:,1],s = 2, label = 'Cluster {}'.format(n))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Devisive Hierarchical of data_h')
plt.legend()
plt.show()

'''
plt.scatter(label_data[0][:,0],label_data[0][:,1],s = 2, label = '1st Cluster')
plt.scatter(label_data[1][:,0],label_data[1][:,1],s = 2, label = '2nd Cluster')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means Clustering of data_kmeans_4')
plt.legend()
plt.show()
'''
