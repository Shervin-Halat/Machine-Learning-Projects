import csv
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

file_address = 'C:/Users/Shervin/Desktop/processed.cleveland.data'

data = []
data2= []
trdata = []
tsdata = []
with open(file_address) as f:
    reader = csv.reader(f)
    data = list(reader)
    data2 = np.array(data)
    print(data2)
    for y in data:
        for z in y:
            if z=='?':
                data.remove(y)
    
    for i in range(0, 2*len(data)//3):
        data[i] = [float(x) for x in data[i]]
        trdata.append(data[i])
    for j in range(2*len(data)//3 , len(data)):
        data[j] = [float(x) for x in data[j]]
        tsdata.append(data[j])
    print(len(tsdata),len(trdata),len(data)) #############
#    print(data)

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

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	print(output_values)
	return prediction
number = int(input())       
prediction = predict_classification(trdata, data[number], 8)
print('Expected %d, Got %d.' % (data[number][-1], prediction))
print(data[number])
