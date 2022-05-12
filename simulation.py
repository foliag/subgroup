#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:04:50 2020

@author: luna
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import source as src


#data generation
def gen_beta_overlap(p,a,npro,k):
    q = a.shape[1]        # 显著变量个数
    weight_overlap = np.random.uniform(size = npro)
    beta = np.zeros((p,k+npro))
    a1 = a.copy()
    for i in range(len(weight_overlap)):  
        a_overlap = a[0,:] * weight_overlap[i] + a[1,:] * (1-weight_overlap[i])
        a1 = np.append(a1,a_overlap).reshape(-1,q)
    beta[0:q,:] = a1.T
    return beta,weight_overlap


def gen_var_overlap(n,p,rho,prop,k,e):
    sigma = np.zeros(p*p).reshape(p,p)
    for i in range(p):
        for j in range(p):
            sigma[i][j]=rho**abs(i-j)
    nprop = np.round(prop*n).astype(int)
    ntmp = np.insert(np.cumsum(nprop),0,0)
    X = np.random.multivariate_normal(np.array([0]*p),sigma,n).T
    Y = np.array(())   
    group = np.array(())
    err = np.array(())
    for i in range(k):
        err_tmp = e * np.random.randn(nprop[i])
        Y = np.append(Y,X[:,ntmp[i]:ntmp[i+1]].T.dot(beta[:,i].T) + err_tmp)
        group = np.append(group, np.ones(nprop[i])*i+1)
        err = np.append(err, err_tmp)
    err_tmp = e * np.random.randn(nprop[-1])
    #overlap
    Y = np.append(Y,(X[:,n-nprop[-1]:]*beta[:,k:]).sum(axis=0) + err_tmp)
    err = np.append(err, err_tmp)
    group = np.append(group, np.ones(nprop[-1])*(k+1))
    #group_pf[i] = np.argmax(weight_pf.iloc[:,i]) + 1
    group = group.astype(int)
    return Y, X, group, err


def gen_beta_nonoverlap(p,a):
    q = a.shape[1]        # number of significant variables
    beta = np.zeros((p,k))
    beta[0:q,:] = a.T
    return beta



def gen_var_nonoverlap(n,p,rho,prop,e):
    sigma = np.zeros(p*p).reshape(p,p)
    for i in range(p):
        for j in range(p):
            sigma[i][j]=rho**abs(i-j)
    nprop = np.round(prop*n).astype(int)
    ntmp = np.insert(np.cumsum(nprop),0,0)
    X = np.random.multivariate_normal(np.array([0]*p),sigma,n).T
    Y = np.array(())   
    group = np.array(())
    err = np.array(())
    for i in range(k):
        err_tmp = e * np.random.randn(nprop[i])
        Y = np.append(Y,X[:,ntmp[i]:ntmp[i+1]].T.dot(beta[:,i].T) + err_tmp)
        group = np.append(group, np.ones(nprop[i])*i+1)
        err = np.append(err, err_tmp)
    group = group.astype(int)
    return Y, X, group, err



###evaluation
def sse_calculate(beta0, weight,X,Y):
    n = X.shape[1]
    beta_indi = beta0.dot(weight)
    resi2 = sum((Y - (beta_indi*X).sum(axis = 0))**2)
    return resi2




def confu(pp_tr,beta_hat):    
    beta_indic = np.concatenate((np.ones(pp_tr),np.zeros(p-pp_tr)))
    beta_hat_indic = np.sign(beta_hat**2)
    fp_n =0; tp_n =0
    for i in range(beta_hat.shape[1]):
        confu = confusion_matrix(beta_indic, beta_hat_indic[:,i])
        fp_n = fp_n + confu[0][1]
        tp_n = tp_n + confu[1][1]
    return fp_n, tp_n
   
    
def confu_indi(beta,beta_hat, weight_hat,np_tr,pp_tr,n):    
    noverlape = beta.shape[1] - k
    beta_true = np.ones((int(n*prop[0]),p))*beta[:,0]
    for kk in range(1,k):
        beta_true = np.vstack((beta_true,np.ones((int(n*prop[kk]),p))*beta[:,kk]))
    beta_true = np.vstack((beta_true,beta[:,k:].T))
    beta_est = beta_hat.dot(weight_hat).T

    beta_true_indic = np.sign(beta_true**2)
    beta_est_indic = np.sign(beta_est**2)
    
    fp_n =0; tp_n =0
    for i in range(beta_est_indic.shape[0]):
        confu = confusion_matrix(beta_est_indic[i,:], beta_true_indic[i,:])
        fp_n = fp_n + confu[0][1]
        tp_n = tp_n + confu[1][1]
    
    fp = fp_n/np_tr/n
    tp = tp_n/pp_tr/n
    return fp, tp



def rmse_multi(beta, beta_hat, weight_hat,n): 
    noverlape = beta.shape[1] - k
    beta_true = np.ones((int(n*prop[0]),p))*beta[:,0]
    for kk in range(1,k):
        beta_true = np.vstack((beta_true,np.ones((int(n*prop[kk]),p))*beta[:,kk]))
    beta_true = np.vstack((beta_true,beta[:,k:].T))
    beta_est = beta_hat.dot(weight_hat)
    rmse = np.sqrt(((beta_true-beta_est.T)**2).sum()/n/p)
    
    return rmse


def l1_loss(group_true,weight_overlap,weight):
    if weight.shape[0] == 2:
        n = len(group_true)
        weight_true = np.zeros((2,n))
        weight_true[0,group_true == 1] = 1
        weight_true[1,group_true == 2] = 1
        weight_true[0:,group_true == 3] = weight_overlap
        weight_true[1:,group_true == 3] = 1 - weight_overlap
        
        loss1 = np.sum(abs(weight-weight_true))/n
    
        weight_true = np.zeros((2,n))
        weight_true[0,group_true == 2] = 1
        weight_true[1,group_true == 1] = 1
        weight_true[0:,group_true == 3] = 1-weight_overlap
        weight_true[1:,group_true == 3] = weight_overlap
        
        loss2 = np.sum(abs(weight-weight_true))/n
        loss = min(loss1,loss2)
    else:
        weight_true1 = np.zeros(n)
        weight_true1[group_true == 1] = 1
        weight_true1[group_true == 3] = weight_overlap
        loss1 = min(abs(weight-weight_true1).sum(axis=1))
        g1 = np.argmin(abs(weight-weight_true1).sum(axis=1))
        
        weight_true2 = np.zeros(n)
        weight_true2[group_true == 2] = 1
        weight_true2[group_true == 3] = 1 - weight_overlap
        loss2 = min(abs(np.delete(weight,g1,axis=0)-weight_true2).sum(axis=1))
        g2 = np.argmin(abs(np.delete(weight,g1,axis=0)-weight_true2).sum(axis=1))
        if g2 <=g1:
            g2 = g2+1
        
        loss_else = np.sum(np.delete(weight,np.array((g1,g2)),axis=0))
        loss = (loss_else+loss1+loss2)/n

    return loss


 





## data generation
p=10
n = 200
rho=0.5
e = 0.5    #variance of error

#s1
k=2
a = np.array([[1,2,3,0,0,0],[0,0,0,-4,-5,-6]])
prop = np.array((0.4,0.4,0.2)) 
npro = int(n*prop[-1])
pp_tr = 6  #positive p
np_tr = p-pp   #negative p

beta,weight_overlap = gen_beta_overlap(p,a,npro,k)
Y,X,group_true, err = gen_var_overlap(n,p,rho,prop,k,e)




#s2
k=2
a = np.array([[1,2,3],[1,-2,-3]])
prop = np.array((0.4,0.4,0.2))
npro = int(n*prop[-1])

beta,weight_overlap = gen_beta_overlap(p,a,npro,k)
Y,X,group_true, err = gen_var_overlap(n,p,rho,prop,k,e)




#s3
k=2
prop = np.array((0.5,0.5))    #proportion of sample size in each group
#prop = np.array((0.3,0.7))
a = np.array([[1,2,3],[1,-2,-3]])   #non-zero coefficients
beta = gen_beta(p,a)
Y,X,group_true, err = gen_var(n,p,rho,prop,e)




#s4
k=2
prop = np.array((0.5,0.5))    #proportion of sample size in each group
#prop = np.array((0.3,0.7))
a = np.array([[1,2,3,0,0,0],[0,0,0,-1,-2,-3]])   #non-zero coefficients
beta = gen_beta_nonoverlap(p,a)
Y,X,group_true, err = gen_var_nonoverlap(n,p,rho,prop,e)


##estimation

#core = multiprocessing.cpu_count()
sig_p, time, resi_sig = src.par_scr(X,Y,m1 = 5, m2 = 5,core = 3)
X_sig = X[sig_p,:]
group_k2, beta_k, evalu_k, ttt_k, group_rep_k = src.rep_kmeans2(X_sig, Y, k, rep_time = 5)
group_est, group_init, center, beta_est, weight_est = src.swkmeans(X, Y, k, lamb=0.1, group_init = group_k2)
weight_up, beta_up, group_up = src.justify(X,Y,weight_est,group_est,beta_est)


rpe = np.sqrt(sse_calculate(beta_est,weight_est,X,Y)/n)
ari = adjusted_rand_score(group_true, group_up)         # for s4-s5
l1loss = l1_loss(group_true,weight_overlap,weight_up)   # for s1-s3
rmse = rmse_multi(beta, beta_up, weight_up,n)
fp,tp = confu(pp_tr,beta_up)

#
sns.heatmap(weight_up.T, cmap="YlGnBu_r")

