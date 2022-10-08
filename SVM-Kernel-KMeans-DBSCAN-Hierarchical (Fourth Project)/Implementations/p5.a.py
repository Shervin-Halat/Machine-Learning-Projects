import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import randint

import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import randint

loc = 'C:/Users/sherw/OneDrive/Desktop/HW4_ml/dataset/data_kmeans_1.txt'
file = open(loc)
data = np.array(list(csv.reader(file)),dtype = float)
data = data.tolist()
print(type(data))
print(type(data[0]))
print(data[0])
'''
data = []
loc = 'C:/Users/sherw/OneDrive/Desktop/HW4_ml/dataset/question5/D31.txt'
file = open(loc)
for line in file:
    data.append(line.split())
data = np.array(data , dtype = float)
x = data[:,0:2]
#np.random.shuffle(x)
x = x.tolist()
y = data[:,-1]

y = y.tolist()
#np.random.shuffle(data)

print(len(y))
'''
plt.scatter(np.array(data)[:,0],np.array(data)[:,1] ,s = 1)
#plt.title('data with initial classes of D31')

plt.show()

def neighbors(point, points, e):
    neigh = []
    for i in points:
        if i != point:
            if np.linalg.norm(np.array(point)- np.array(i)) <= e:
                neigh.append(i)
    return neigh
            
def dbscan(dataset, eps, mp):
    maindata = list(dataset)
    core, noise, border, tovisit = [] , [] , [], []
    count = 0
    while True:
        count+= 1
        #print(count)
        print('#######',len(dataset))
        tovisit = []
        index = randint(0,len(dataset)-1)
        init = dataset[index]
        nei = neighbors(init , maindata, eps)
        if len(nei) >= mp:
            #print(len(dataset))
            del dataset[dataset.index(init)]
            #print(len(dataset))
            core.append([])
            border.append([])
            core[-1].append(init)
            for i in nei:
                tovisit.append(i)
            for j in tovisit:
                if len(neighbors(j,maindata,eps)) >= mp:
                    core[-1].append(j)
                    if j in dataset:
                        #print(len(dataset))
                        del dataset[dataset.index(j)]
                        #print(len(dataset))
                    for ne in neighbors(j,dataset,eps):
                        if ne not in tovisit:
                            tovisit.append(ne)
                elif len(neighbors(j,maindata,eps)) < mp:
                    border[-1].append(j)
                    
                    if j in dataset:
                        #print(len(dataset))
                        del dataset[dataset.index(j)]
                        #print(len(dataset))
                    for ne in neighbors(j,dataset,eps):
                        if ne not in tovisit:
                            tovisit.append(ne)
                    #del dataset[dataset.index(j)]

        ############## noise handler      
        elif len(nei) < mp:
            f = len(nei)
            counter = 0
            for i in nei:
                if len(neighbors(i,maindata,eps)) < mp:
                    counter += 1
            if counter == f:
                noise.append(init)
                del dataset[index]
                        
        if not list(dataset):
            break
    for i in border:
        for j in i:
            if j in noise:
                noise.remove(j)
    
    clusters = []
    print('core2',len(core))
    for i in range(len(core)):
        clusters.append([])
        clusters[i] = core[i] + border[i]
    
    return clusters, noise
#epsilon , minpoints
cl, n = dbscan(data,1.2,6)
cl = np.array(cl)

for i in range(len(cl)):
    cl[i] = np.array(cl[i])

print(n)
n = np.array(n)

m = 0
for i in cl:
    m+=1
    plt.scatter(i[:,0],i[:,1], s = 2 , label = 'cluster{}'.format(m))

if len(n) > 0:
    plt.scatter(n[:,0],n[:,1], s = 2 , label = 'noises')

plt.title('DBSCAN clustering for database \'D31\'')
plt.legend()
plt.show()

