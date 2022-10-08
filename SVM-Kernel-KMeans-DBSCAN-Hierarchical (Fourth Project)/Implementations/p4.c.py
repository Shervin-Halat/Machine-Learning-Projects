import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import sample
from random import randint
from math import sqrt
from scipy import spatial

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


def mindist(cl1,cl2):
    min_dist = np.min(spatial.distance.cdist(cl1, cl2))
    return min_dist


def maxdist(cl1,cl2):
    max_dist = np.max(spatial.distance.cdist(cl1, cl2))
    return max_dist


def avrdist(cl1,cl2):
    avr_dist = sum([np.linalg.norm(i-j) for i in cl1 for j in cl2]) / (len(cl1)*len(cl2))
    return avr_dist


def sl(clus):
    n = len(clus)
    disdict = {}
    for i in clus:
        for j in clus:
            if not(np.array_equal(i,j)):
                disdict[([np.array_equal(i,x) for x in clus].index(True),[np.array_equal(j,x) for x in clus].index(True))] \
                                              = mindist(i,j)
    (a,b) = min(disdict, key=disdict.get)
    return a,b

    
def cl(clus):
    n = len(clus)
    disdict = {}
    for i in clus:
        for j in clus:
            if not(np.array_equal(i,j)):
                disdict[([np.array_equal(i,x) for x in clus].index(True),[np.array_equal(j,x) for x in clus].index(True))] \
                                              = maxdist(i,j)
    (a,b) = min(disdict, key=disdict.get)
    return a,b


def al(clus):
    n = len(clus)
    disdict = {}
    for i in clus:
        for j in clus:
            if not(np.array_equal(i,j)):
                disdict[([np.array_equal(i,x) for x in clus].index(True),[np.array_equal(j,x) for x in clus].index(True))] \
                                              = avrdist(i,j)
    (a,b) = min(disdict, key=disdict.get)
    return a,b


def hier_agg(dataset , n):
    clusters = kmeans(dataset,8)
    
    while True:
        ########
        temp = list(clusters)
        for i in range(len(temp)):
            temp[i] = np.array(temp[i])
        for i in temp:
            plt.scatter(i[:,0],i[:,1],s = 2)
        plt.show()
        ########
        c1,c2 = al(clusters)
        temp_clust = np.concatenate((clusters[c1], clusters[c2]), axis=0)
        
        indexes = [c1,c2]
        for index in sorted(indexes, reverse=True):
            del clusters[index]

        clusters.append(temp_clust)
        if len(clusters) == n:
            break
    for i in range(len(clusters)):
        clusters[i] = np.array(clusters[i])
    return clusters

c = hier_agg(data,4)

n = 0
for i in c:
    n += 1
    plt.scatter(i[:,0],i[:,1],s = 2, label = 'Cluster {}'.format(n))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Agglomerative Hierarchical of data_h')
plt.legend()
plt.show()

