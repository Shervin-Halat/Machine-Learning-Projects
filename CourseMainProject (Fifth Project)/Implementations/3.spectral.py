import pandas
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import csv
from random import randint

loc = 'C:/Users/sherw/OneDrive/Desktop/ML_Project/Dataset/Clustering/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls'
data = pandas.read_excel(loc,'Training_Data')
data = np.array([ data[column].values for column in data.columns]).T
data = np.array(data[:,0:5],dtype = float)
print(data.shape)

c = []
error = []
error2 = []
nn = range(1,11)
#for n in nn:
sc = SpectralClustering(n_clusters = 6)
sc = sc.fit(data)
p = sc.labels_
error = []
e = 0
'''
    for i in range(n):
        error.append([])
        for j in range(len(p)):
            if p[j] == i:
                error[i].append(data[i])
    error = np.array(error)
    for k in range(len(error)):
        for l in error[:,k]:
            e += abs(l - (error[:,k].mean))
        e = e/len(error[:,k])
    error2.append(e)
'''

#n = error2.index(min(error2))
print(p)

with open( 'C:/Users/sherw/OneDrive/Desktop/ML_Project/3.dbscan.csv', 'w',newline = '') as h:
    writer = csv.writer(h)
    for i in p:
        writer.writerow([int(i)])



        
    
