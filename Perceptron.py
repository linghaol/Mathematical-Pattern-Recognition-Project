# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:32:48 2017

@author: LLH
"""
import numpy as np
from random import sample

# 2-class Perceptron learning classifier
def perceptron(dat, increment=1):
    '''
    Perceptron learning for 2-class classification(class label must be 1 and -1!)

    dat: data matrix, augmented & reflected, the last element is class label
    increment: default value = 1
    
    return weight vector w
    '''
    # default max_iterations
    max_iter = 20000
    
    # initialize weight vector: w   
    w = np.ones(len(dat[0])-1)
    
    # main procedure (containing reflecting progress by multiplying label 1 or -1)
    n = len(dat)
    for i in range(max_iter):
        shuf_order = sample(range(n),n)
        count = 0
        for j in shuf_order:
            if np.inner(dat[j,:-1],w)*dat[j,-1] < 0:
                w = w + increment*(dat[j,:-1]*dat[j,-1])
                break
            else:
                count += 1
        if count == n:
            print('Converged!')
            break
    if i == max_iter - 1:
        print('Reach maximum iteration limit: ',max_iter) 
    
    # calculate predicted label
    pre_label = []
    for i in dat:
        if sum([np.inner(a,b) for a,b in zip(i[:-1],w)]) > 0:
            pre_label.append(1)
        else:
            pre_label.append(-1)
    
    return w, pre_label
