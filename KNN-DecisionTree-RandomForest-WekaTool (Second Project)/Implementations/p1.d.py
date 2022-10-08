import csv
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

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
		dist = manhattan_distance(test_row, train_row)
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
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(dataset)):
        if (dataset[i][-1] == 1 and predict_classification(trdata, dataset[i], k) == 1):
            tp += 1
        elif (dataset[i][-1] == 0 and predict_classification(trdata, dataset[i], k) == 0):
            tn += 1
        elif (dataset[i][-1] == 0 and predict_classification(trdata, dataset[i], k) == 1):
            fp += 1
        elif (dataset[i][-1] == 1 and predict_classification(trdata, dataset[i], k) == 0):
            fn += 1
    return tp,tn,fp,fn

#main:
k_values = [1,2,3,4,5,6,7,10,15]
accuracy = []
for k in k_values:
    acc = precision(tsdata,k)#here accuracy is just a name to use in next line
    accuracy.append(acc)
    print('accuracy of KNN algorithm for K = %i is %f' %(k , acc))

best_k = k_values[accuracy.index(max(accuracy))]

print('so the best K-value would be %i' %best_k)

print('accuracy of training data for K-value of %s is %f \naccuracy of test data for K-value of %s is %f'\
    %(best_k , precision(trdata, best_k) , best_k , precision(tsdata,best_k)))

print('tp,tn,fp,fn\n',con_mat(trdata , best_k) , con_mat(tsdata , best_k) , sep ='')
