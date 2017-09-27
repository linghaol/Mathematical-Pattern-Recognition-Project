# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:39:49 2017

@author: LLH
"""
# EE559 Project - Linghao Li

import numpy as np
from pandas import read_csv
from random import sample
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report as report


#------------------------------Pre-Processing---------------------------------#
# input data
data = read_csv('W:/Python/test file/EE559 Project/Proj_dataset_1.csv',
                 keep_default_na = False)

# augment feature space
data['Unnamed: 0'] = np.ones(len(data['Unnamed: 0'])).tolist()

# numerically categorize feature
cat_Sex = {'male':1, 'female':0}
cat_Hou = {'own':3, 'rent':2, 'free':1}
cat_Sca = {'quite rich':5, 'rich':4, 'moderate':3, 'little':2, 'NA':1}
cat_Pur = {'business':            [1,0,0,0,0,0,0,0],
           'car':                 [0,1,0,0,0,0,0,0],
           'domestic appliances': [0,0,1,0,0,0,0,0],
           'education':           [0,0,0,1,0,0,0,0],
           'furniture/equipment': [0,0,0,0,1,0,0,0],
           'radio/TV':            [0,0,0,0,0,1,0,0],
           'repairs':             [0,0,0,0,0,0,1,0],
           'vacation/others':     [0,0,0,0,0,0,0,1]}
cat_Cla = {1:1, 2:-1}

data['Sex']              = [cat_Sex[i] for i in data['Sex']]
data['Housing']          = [cat_Hou[i] for i in data['Housing']]
data['Saving accounts']  = [cat_Sca[i] for i in data['Saving accounts']]
data['Checking account'] = [cat_Sca[i] for i in data['Checking account']]
data['Purpose']          = [cat_Pur[i] for i in data['Purpose']]
data['Class']            = [cat_Cla[i] for i in data['Class']]


#---------------------------Normalization (Min-Max)---------------------------#
for i in ['Age','Credit amount', 'Duration']:
    maxi = np.max(data[i])
    mini = np.min(data[i])
    data[i] = (data[i]-mini)/(maxi-mini)


#-----------------------Training set and Testing set--------------------------#
# Testing set: 20%, Validation set: 20%, Training set: 60% (class preserved)
n = len(data)
ind_c1 = list(data.loc[data['Class']==1].index)
ind_c2 = list(data.loc[data['Class']==-1].index)
n_c1 = len(ind_c1)
n_c2 = len(ind_c2)
Te_ind = sample(ind_c1,int(n_c1*0.2))+sample(ind_c2,int(n_c2*0.2))
ind_oth_c1 = list(set(ind_c1).difference(set(Te_ind)))
ind_oth_c2 = list(set(ind_c2).difference(set(Te_ind)))
n_oth_c1 = len(ind_oth_c1)
n_oth_c2 = len(ind_oth_c2)
Va_ind = sample(ind_oth_c1,int(n_oth_c1*0.25))+sample(ind_oth_c2,int(n_oth_c2*0.25))
Tr_ind = list(set(ind_oth_c1).union(set(ind_oth_c2)).difference(set(Va_ind)))

mat = np.array([list(i[:-2])+i[-2]+[i[-1]] for i in data.values])
Testset   = mat[Te_ind]
Validset  = mat[Va_ind]
Trainset  = mat[Tr_ind]


#-----------------------PCA for Dimensionality Reduction----------------------#
mypca = PCA(n_components=5)
mypca.fit(Trainset[:,1:-1])
np.sum(mypca.explained_variance_ratio_)

# drawing figure
ratio = mypca.explained_variance_ratio_
index = np.arange(0,4,0.8)
width = 0.4
with plt.style.context('ggplot'):
    plt.bar(index, ratio, width)
    plt.xticks(index+width/2, ['PC1','PC2','PC3','PC4','PC5'], fontsize=15)
    plt.title('Explained Variance Ratio', fontsize=17)
    plt.yticks(fontsize=15)
    plt.ylabel('Pencentage(%)', fontsize=17)

Tr_pca = np.c_[Trainset[:,0], mypca.transform(Trainset[:,1:-1]), Trainset[:,-1]]
Va_pca = np.c_[Validset[:,0], mypca.transform(Validset[:,1:-1]), Validset[:,-1]]
Te_pca = np.c_[Testset[:,0], mypca.transform(Testset[:,1:-1]), Testset[:,-1]]


#-----------------------Method 1: Perceptron Learning-------------------------#
from Perceptron import *

# Original features
w_p, pre_p_train = perceptron(Trainset, increment=2)

pre_p_valid = []
for i in Validset:
    if np.inner(i[:-1],w_p) > 0:
        pre_p_valid.append(1.0)
    else:
        pre_p_valid.append(-1.0)
print(report(Validset[:,-1], pre_p_valid, target_names=['Class 1', 'Class 2']))

pre_p_test = []
for i in Testset:
    if np.inner(i[:-1],w_p) > 0:
        pre_p_test.append(1.0)
    else:
        pre_p_test.append(-1.0)
print(report(Testset[:,-1], pre_p_test, target_names=['Class 1', 'Class 2']))

# With PCA
w_p_pca, pre_p_train_pca = perceptron(Tr_pca, increment=2)

pre_p_valid_pca = []
for i in Va_pca:
    if np.inner(i[:-1],w_p_pca) > 0:
        pre_p_valid_pca.append(1.0)
    else:
        pre_p_valid_pca.append(-1.0)
print(report(Va_pca[:,-1], pre_p_valid_pca, target_names=['Class 1', 'Class 2']))

pre_p_test_pca = []
for i in Te_pca:
    if np.inner(i[:-1],w_p_pca) > 0:
        pre_p_test_pca.append(1.0)
    else:
        pre_p_test_pca.append(-1.0)
print(report(Te_pca[:,-1], pre_p_test_pca, target_names=['Class 1', 'Class 2']))


#------------------------------Method 2: SVM----------------------------------#
from svmutil import *
prob = svm_problem(Trainset[:,-1], Trainset[:,:-1].tolist())
para = svm_parameter('-t 1 -c 15 -q')
model = svm_train(prob, para)

labs_valid, acc, vals = svm_predict(Validset[:,-1], Validset[:,:-1].tolist(), model, '-q')
print(report(Validset[:,-1], labs_valid, target_names=['Class 1', 'Class 2']))

labs_test, acc, vals = svm_predict(Testset[:,-1], Testset[:,:-1].tolist(), model, '-q')
print(report(Testset[:,-1], labs_test, target_names=['Class 1', 'Class 2']))

# with PCA
prob_pca = svm_problem(Tr_pca[:,-1], Tr_pca[:,:-1].tolist())
para_pca = svm_parameter('-t 2 -c 500 -q')
model_pca = svm_train(prob_pca, para_pca)

labs_valid_pca = svm_predict(Va_pca[:,-1], Va_pca[:,:-1].tolist(), model_pca, '-q')[0]
print(report(Va_pca[:,-1], labs_valid_pca, target_names=['Class 1', 'Class 2']))

labs_test_pca = svm_predict(Te_pca[:,-1], Te_pca[:,:-1].tolist(), model_pca, '-q')[0]
print(report(Te_pca[:,-1], labs_test_pca, target_names=['Class 1', 'Class 2']))


#------------------Method 3: Bayes Min. Error based on KNN--------------------#
# Original features
BME = KNN(n_neighbors=24)
BME.fit(Trainset[:,1:-1], Trainset[:,-1])

pre_knn_valid = BME.predict(Validset[:,1:-1])
print(report(Validset[:,-1], pre_knn_valid, target_names=['Class 1', 'Class 2']))

pre_knn_test = BME.predict(Testset[:,1:-1])
print(report(Testset[:,-1], pre_knn_test, target_names=['Class 1', 'Class 2']))

# With PCA
BME_pca = KNN(n_neighbors=24)
BME_pca.fit(Tr_pca[:,1:-1], Tr_pca[:,-1])

pre_knn_valid_pca = BME_pca.predict(Va_pca[:,1:-1])
print(report(Va_pca[:,-1], pre_knn_valid_pca, target_names=['Class 1', 'Class 2']))

pre_knn_test_pca = BME_pca.predict(Te_pca[:,1:-1])
print(report(Te_pca[:,-1], pre_knn_test_pca, target_names=['Class 1', 'Class 2']))





