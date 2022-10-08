import numpy as np
import csv
from math import sqrt
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

pdf = scipy.stats.norm.pdf
data = []

with open('C:/Users/sherw/Desktop/ML3/wine.csv') as file:
    reader = csv.reader(file)
    data = np.array(list(reader),dtype = float)
    #data = list(reader)

##############################################################
# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = int(vector[0])
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# Calculate prior probs
def prior_probs(dataset):
        class_prob = {}
        separated = separate_by_class(dataset)
        n = len(data)
        for i in list(separated.keys()):
                class_prob[i] = len(list(separated[i])) / n
        return class_prob

# Calculate the mean, standard deviation 
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset  
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[0])
	return summaries

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# Predict class of row x
def class_predict(test_row , dataset):
        class_prob = prior_probs(dataset)
        summaries = summarize_by_class(dataset)
        classes = list(summaries.keys())
        predictions = []
        for i in classes:
                ln_pdf = 0
                prob = 0
                for j in range(len(summaries[i])):
                        ln_pdf += np.log(pdf(test_row[j+1] , summaries[i][j][0] , summaries[i][j][1]))
                prob = ln_pdf + np.log(class_prob[i])
                predictions.append([i,prob])
        predictions = sorted(predictions , key = lambda x: x[1] , reverse = True)
        return predictions[0][0]

def acc(testset , trainset):
        counter = 0
        for row in testset:
                predict = class_predict(row , trainset)
                if predict == int(row[0]):  ####
                        counter += 1
        accu = counter/len(testset)
        return accu

#instead of class_predict, each row all classes probs:
def row_class_prob(test_row , dataset):
        class_prob = prior_probs(dataset)
        summaries = summarize_by_class(dataset)
        classes = list(summaries.keys())
        predictions = []
        for i in classes:
                ln_pdf = 0
                prob = 0
                for j in range(len(summaries[i])):
                        ln_pdf += np.log(pdf(test_row[j+1] , summaries[i][j][0] , summaries[i][j][1]))
                prob = ln_pdf + np.log(class_prob[i])
                predictions.append([i,prob])
                predictions = sorted(predictions , key = lambda x: x[0])
        return predictions

####################################MAIN##################################
# 2 positive and 1 negative
np.random.shuffle(data)
new_data = [list(rows) for rows in data if int(rows[0]) == 1 or int(rows[0]) == 2]

train_data = new_data[0 : int(0.7*len(new_data))]
test_data = new_data[int(0.7*len(new_data)) : len(new_data)]
threshold = np.linspace(-10 , 0 , 300)

n_p , n_n = 0 , 0
for row in test_data:
        if int(row[0]) == 1:
                n_n += 1
        else:
                n_p += 1
                
TPR,FPR = [],[]
for i in threshold:
        t_p,f_p = 0,0
        for row in test_data:
                probs = row_class_prob(row , train_data)
                if probs[1][1] - probs[0][1] >= i :
                        if int(row[0]) == int(probs[1][0]):
                                t_p += 1
                        else:
                                f_p += 1
        TPR.append(t_p / n_p)
        FPR.append(f_p / n_n)

plt.plot(FPR , TPR , label = 'classifier')
plt.plot([0,1],[0,1], label = 'diagonal')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC(naive bayes)')
plt.legend()
plt.show()
