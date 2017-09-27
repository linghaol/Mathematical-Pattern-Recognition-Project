# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 05:55:46 2017

@author: LLH
"""

# EE 559 Project - Random Forest version

from pandas import read_csv, DataFrame, concat
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import classification_report as report
from random import sample

#------------------------------Pre-Processing---------------------------------#
# input data
data = read_csv('W:/Python/test file/EE559 Project/Proj_dataset_1.csv',
                 keep_default_na = False)
Y = data['Class']
del data['Unnamed: 0']
del data['Class']

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

for i in cat_Sex.items():
    data.loc[data['Sex']==i[0],'Sex'] = i[1]
    
for i in cat_Hou.items():
    data.loc[data['Housing']==i[0],'Housing'] = i[1]
    
for i in cat_Sca.items():
    data.loc[data['Saving accounts']==i[0],'Saving accounts'] = i[1]
    data.loc[data['Checking account']==i[0],'Checking account'] = i[1]

data['Purpose'] = [cat_Pur[i] for i in data['Purpose']]

X = concat([data, DataFrame(data['Purpose'].tolist(), columns=cat_Pur.keys(), dtype=object)], axis=1)

del X['Purpose']
del data

#-----------------------Training set and Testing set--------------------------#
# Training set: 70%, Testing set: 30% (class preserved)
n = len(Y)
ind_c1 = list(Y.loc[Y==1].index)
ind_c2 = list(Y.loc[Y==2].index)
n_c1 = len(ind_c1)
n_c2 = len(ind_c2)
Tr_ind = sample(ind_c1,int(n_c1*0.7))+sample(ind_c2,int(n_c2*0.7))
Te_ind = sample(ind_c1,int(n_c1*0.3))+sample(ind_c2,int(n_c2*0.3))

TrainSet = X.loc[Tr_ind]
TrainClass = Y.loc[Tr_ind]

TestSet = X.loc[Te_ind]
TestClass = Y.loc[Te_ind]

del X, Y

#------------------------------Random Forest----------------------------------#
clf = RFC(n_estimators=20)
clf = clf.fit(TrainSet, TrainClass)

#----------------------------Prediction & Report------------------------------#
PredictClass = clf.predict(TestSet)
print(report(TestClass, PredictClass, target_names=['Class 1', 'Class 2']))



