import csv
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn import datasets
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

digits.images = digits.images.reshape(digits.images.shape[0],\
                                     digits.images.shape[1]*\
                                     digits.images.shape[2])
digits_images = digits.images
digits_target = digits.target

#from sklearn.model_selection import train_test_split

x_train,x_test,z_train,z_test = train_test_split(digits.images\
                            ,digits.target,test_size = 0.2)
x_train,x_test,y_train,y_test = train_test_split(x_train\
                            ,z_train,test_size = 0.25)


#from sklearn.neighbors import KNeighborsClassifier

k = [k+1 for k in range(20)]
accuracy = []
for i in k:
    clf = KNeighborsClassifier(n_neighbors = i)\
                            .fit(x_train,y_train)
    accuracy.append(accuracy_score(y_test,clf.predict(x_test)))
    #from sklearn.metrics import accuracy_score
    print("accuracy of test set using K-value of %i set is %f" \
      %(i,accuracy[i-1]))
print("the best K-value by comparison is 1")
