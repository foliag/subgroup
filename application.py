#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 17:24:29 2021

@author: luna
"""



import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
#from numba import jit
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, confusion_matrix
from scipy import stats
import warnings
import multiprocessing
warnings.filterwarnings("ignore")

import src


##############CCLE

#### DATA IMPORT

y = pd.read_csv("CCLE_Y.csv",index_col = 0).iloc[:,14]   #PF2341066 as response
x = pd.read_csv("CCLE_X.csv",index_col = 0)


#pre-screening result
f = open("screening_genes.txt",'r')     #genes for the top 120 submodels 
sig_name = []
i=0
while True:
    line = f.readline()
    if not line:
        break
    line = line.strip('\n')
    sig_name.append(line)

sig_p =  np.sort(np.where(x.index.isin(sig_name))[0])
x = x.iloc[sig_p,:]


#cell line type
celltype = pd.read_csv("cell_line_type.csv",index_col = 0)  
group_true = celltype['group']



#### PREPROCESSING
#drop NA
x.drop(x.columns[np.where(np.isnan(y))[0]],axis = 1,inplace = True)
group_true.drop(group_true.index[np.where(np.isnan(y))[0]],inplace = True)
celltype.drop(celltype.index[np.where(np.isnan(y))[0]],inplace = True)
y.dropna(axis = 0,inplace = True)  

p, n = x.shape
sig_name = x.index.tolist()


Y = preprocessing.scale(y)
X = preprocessing.scale(x.T).T


#### ESTIMATION
scr_p, time, resi_sig = src.par_scr(X,Y,m1 = 5, m2 = 5,core = 3)
X_scr = X.iloc[scr_p,:]
group_k1, beta_k, evalu_k, ttt_k, group_rep_k = src.rep_kmeans2(X, Y, 4, rep_time = 5)
group_est, group_init, center_est, beta_est, weight_est = src.swkmeans_app(X, Y, 4, lamb=0.001, group_init = group_k1)


###################LUNG CANCER


#### DATA IMPORT
luad_x = pd.read_csv("GE_LUAD.csv",sep = ';',index_col = 0)
lusc_x = pd.read_csv("GE_LUSC.csv",sep = ';',index_col = 0)

luad_y = pd.read_csv("LUAD_clin.csv",sep = ';',index_col = 0)
lusc_y = pd.read_csv("LUSC_clin.csv",sep = ';',index_col = 0)


luad = luad_y.merge(luad_x,left_index = True, right_index = True)
lusc = lusc_y.merge(lusc_x,left_index = True, right_index = True)

lung = pd.concat([luad,lusc])

##clinical information

luad_clin = pd.read_csv("TCGA-LUAD-biolinks-clinical.csv",sep=';')
lusc_clin = pd.read_csv("TCGA-LUSC-biolinks-clinical.csv",sep=';')

luad_clin.dropna(axis=1,how='all',inplace = True)
lusc_clin.dropna(axis=1,how='all',inplace = True)

lung_clin = pd.concat([lusc_clin,luad_clin])

lung_clin['disease_ind'] = 1
lung_clin.loc[lung_clin['disease'] == 'LUAD','disease_ind'] = 2

lung_clin['status'] = 0
lung_clin.loc[lung_clin['vital_status'] == 'Dead','status'] = 1


##anova result
f = open("lusc_sig_gene.txt",'r')
lines = []
while True:
    line = f.readline()
    if not line:
        break
    line = line.strip().split("\t")
    lines.append(line)
    
sig_lusc = pd.DataFrame(lines).iloc[1:,0]

f = open("luad_sig_gene.txt",'r')
lines = []
while True:
    line = f.readline()
    if not line:
        break
    line = line.strip().split("\t")
    lines.append(line)
    
sig_luad = pd.DataFrame(lines).iloc[1:,0]
sig = set(sig_luad) | set(sig_lusc)

X_anova = lung.iloc[:,lung.columns.isin(sig)]
X = preprocessing.scale(X_anova).T


p,n = X.shape
group_true = np.ones(n)
group_true[lung.iloc[:,1] == 'LUSC'] = 2


#### ESTIMATION
scr_p, time, resi_sig = src.par_scr(X,Y,m1 = 5, m2 = 5,core = 3)
X_scr = X.iloc[scr_p,:]
group_k1, beta_k, evalu_k, ttt_k, group_rep_k = src.rep_kmeans2(X, Y, 2, rep_time = 5)
group_est, group_init, center_est, beta_est, weight_est = src.swkmeans_app(X, Y, 2, lamb=0.001, group_init = group_k1)




