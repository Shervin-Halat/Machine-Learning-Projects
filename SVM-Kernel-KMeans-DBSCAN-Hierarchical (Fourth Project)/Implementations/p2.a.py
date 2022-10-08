import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report

from sklearn.preprocessing import normalize

loc = 'C:/Users/sherw/OneDrive/Desktop/LSVT_voice_rehabilitation.xlsx'
data = pandas.read_excel(loc,'Data' )
label = pandas.read_excel(loc,'Binary response' )


y = label['Binary class 1=acceptable, 2=unacceptable'].values
x = np.array([ data[column].values for column in data.columns ]).T

#np.random.shuffle(x)
print(x.shape)

xnorm = normalize(x, axis=0)
print(xnorm.shape)
print(np.linalg.norm(xnorm[:,7]))

'''
print('Linear Kernel\n\n\n')
svm_lk = SVC(kernel= 'linear')
svm_lk.fit(xnorm,y)
y_lk = svm_lk.predict(xnorm)
score_lk = cross_validate(svm_lk, xnorm, y,scoring = ['f1','accuracy'], cv = 10)
print(y_lk)
print('F1-score of each 10-folds: \n', score_lk['test_f1'],'\ntotal F1-score ={:.2f}'\
      .format(np.mean(score_lk['test_f1'])))
print('accuracy of each 10-folds: \n', score_lk['test_accuracy'],'\ntotal accuracy ={:.2f}'\
      .format(np.mean(score_lk['test_accuracy'])))
'''

r_values = [0.1,0.05,4]
d_values = [3,5,7]
for r in r_values:
    d = 2.5
    svm_pk = SVC(kernel= 'poly' , coef0 = r, degree = d)
    svm_pk.fit(x,y)
    y_pk = svm_pk.predict(x)
    score_pk = cross_validate(svm_pk, xnorm, y,scoring = ['f1','accuracy'], cv = 10)
    print(y_pk)
    print('For r-value of {} and d value of {} we have:'.format(r,d))
    print('F1-score of each 10-folds: \n', score_pk['test_f1'],'\ntotal F1-score ={:.2f}'\
      .format(np.mean( score_pk['test_f1'])))
    print('accuracy of each 10-folds: \n', score_pk['test_accuracy'],'\ntotal accuracy ={:.2f}'\
      .format(np.mean(score_pk['test_accuracy'])))

'''
print('RBF\n\n\n')
g_values = [0.01,1.08,7]
for g in g_values:
    svm_rbf = SVC(kernel= 'rbf', gamma = g)
    svm_rbf.fit(xnorm,y)
    y_rbf = svm_rbf.predict(xnorm)
    score_rbf = cross_validate(svm_rbf, xnorm, y,scoring = ['f1','accuracy'], cv = 10)
    print(y_rbf)
    print('For gamma-value of {} we have:'.format(g))
    print('F1-score of each 10-folds: \n', score_rbf['test_f1'],'\ntotal F1-score ={:.2f}'\
      .format(np.mean(score_rbf['test_f1'])))
    print('accuracy of each 10-folds: \n', score_rbf['test_accuracy'],'\ntotal accuracy ={:.2f}'\
      .format(np.mean(score_rbf['test_accuracy'])))
'''
'''
print('\n\n\nSIGMOID')
coef0 = [1.5,0.05,0.005]
for r in coef0:
    svm_sig = SVC(kernel= 'sigmoid', coef0 = r)
    svm_sig.fit(xnorm,y)
    y_sig = svm_sig.predict(xnorm)
    score_sig = cross_validate(svm_sig, xnorm, y,scoring = ['f1','accuracy'], cv = 10)
    print(y_sig)
    print('For r-value of {} we have:'.format(r))
    print('F1-score of each 10-folds: \n', score_sig['test_f1'],'\ntotal accuracy ={:.2f}'\
      .format(np.mean(score_sig['test_f1'])))
    print('accuracy of each 10-folds: \n', score_sig['test_accuracy'],'\ntotal accuracy ={:.2f}'\
      .format(np.mean(score_sig['test_accuracy'])))
'''
