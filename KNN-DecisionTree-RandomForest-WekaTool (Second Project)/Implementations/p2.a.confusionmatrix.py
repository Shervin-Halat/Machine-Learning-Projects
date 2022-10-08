import csv
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn import datasets
from PIL import Image

digits = datasets.load_digits()
#print(digits.target)

#plt.imshow(digits.images[5],cmap='gray')
#plt.show()

digits.images = digits.images.reshape(digits.images.shape[0],\
                                     digits.images.shape[1]*\
                                     digits.images.shape[2])
digits_images = digits.images
digits_target = digits.target
data = np.c_[digits_images , digits_target]

trdata , evdata , tsdata = [] , [] , []
for i in range(0, 3*len(data)//5):
    trdata.append(data[i])
for i in range(3*len(data)//5, 4*len(data)//5):
    evdata.append(data[i])
for i in range(4*len(data)//5, 5*len(data)//5):
    tsdata.append(data[i])

print(len(trdata), len(evdata), len(tsdata) , len(data))
train, validate, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])
print(len(train),len(validate),len(test))
'''
file_address = 'C:/Users/Shervin/Desktop/processed.cleveland.data'

data = []
data2 = []
trdata = []
tsdata = []
with open(file_address) as f:
    reader = csv.reader(f)
    data = list(reader)
    for y in data:
        for z in y:
            if z=='?':
                data.remove(y)
    for i in data:
        if int(i[13]) >= 1:
            i[13] = 1
        else:
            i[13] = 0
            
    for i in range(0, 2*len(data)//3):
        data[i] = [float(x) for x in data[i]]
        trdata.append(data[i])
    for j in range(2*len(data)//3 , len(data)):
        data[j] = [float(x) for x in data[j]]
        tsdata.append(data[j])
    print(len(tsdata),len(trdata),len(data))

##############################################################
# Find the min and max values for each column
def rescale_data(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]) 
rescale_data(tsdata)
rescale_data(trdata)
################################################################
'''


# calculate the Euclidean distance between two vectors        
def euclidean_distance(row1, row2):
	distance = 0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# calculate the Manhattan distance between two vectors
def manhattan_distance(row1, row2):
    distance = 0
    for i in range(len(row1)-1):
        distance += abs(row1[i] - row2[i])
    return distance
    
# calculate the Chebyshev distance between two vectors   
def chebyshev_distance(row1, row2):
    distances = []
    for i in range(len(row1)-1):
        distances.append(abs(row1[i]-row2[i]))
    distance = max(distances)
    return distance
        
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
#neighbors consist of K closest rows to test-row

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)   ###
	#print(output_values)
	return prediction

'''
number = int(input())       
prediction = predict_classification(trdata, data[number], 8)
print('Expected %d, Got %d.' % (data[number][-1], prediction))
print(data[number])
'''

#Precision dataset
def precision(dataset , k):
    counter = 0
    for i in range(len(dataset)):
        if (dataset[i][-1] == predict_classification(trdata, dataset[i], k)):
            counter += 1
    acc = counter / len(dataset)
    return acc

#Confusion_Matrix
def con_mat(dataset , k):
    conf_matr = np.zeros((10, 10))
    counter = 0
    for i in range(len(dataset)):
        actual = int(dataset[i][-1])
        predicted = int(predict_classification(trdata, dataset[i], k))
        conf_matr[actual][predicted] += 1
        counter += 1
    return conf_matr , counter

'''
#main:
k_values = [i+1 for i in list(range(30))]
accuracy = []
for k in k_values:
    acc = precision(evdata,k)#here accuracy is just a name to use in next line
    accuracy.append(acc)
    print('accuracy of KNN algorithm for K = %i is %f' %(k , acc))

best_k = k_values[accuracy.index(max(accuracy))]
'''
best_k = 1
print('so the best K-value would be %i' %best_k)
'''
print('accuracy of training data for K-value of %s is %f \naccuracy of evaluation data for K-value of %s is %f\
      \naccuracy of test data for K-value of %s is %f'\
      %(best_k , precision(trdata, best_k) , best_k , precision(evdata,best_k) , best_k , precision(tsdata, best_k)))
'''

res , count = con_mat(tsdata , 1)
print(res , count)

