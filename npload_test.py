#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:54:58 2018

@author: icedeath
"""

import numpy as np
with np.load('test_shifted10_12.npz') as data:
    x_train = data['x_train']
    '''
    y_train = data['y_train'],
    x_test = data['x_test'], 
    y_test = data['y_test'],
    x_train0 = data['x_train0'], 
    x_train1 = data['x_train1'],
    x_test0 = data['x_test0'], 
    x_test1 = data['x_test1'],
    y_train1 = data['y_train1'], 
    y_test1 = data['y_test1']
    '''