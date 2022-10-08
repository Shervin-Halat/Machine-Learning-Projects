import pandas
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv

loc = 'C:/Users/sherw/OneDrive/Desktop/ML_Project/Dataset/Clustering/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls'
data = pandas.read_excel(loc,'Training_Data')
data = np.array([ data[column].values for column in data.columns]).T
data = np.array(data[:,0:5],dtype = float)
print(data.shape)

## elbow method for optimal number of clusters
sse = []
n = range(1,11)
for k in K:
    km = KMeans(n_clusters = k)
    #print(km.inertia_)
    km = km.fit(data)
    sse.append(km.inertia_)
plt.xticks(K)
plt.plot(K, sse)
plt.title('SSE based on K value')
plt.show()

k = 7
km = KMeans(n_clusters = k).fit(data)
p = km.labels_

with open( 'C:/Users/sherw/OneDrive/Desktop/ML_Project/3.kmeans.csv', 'w',newline = '') as h:
    writer = csv.writer(h)
    for i in p:
        writer.writerow([int(i)])


        
    
