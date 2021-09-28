#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:10:22 2020

@author: luna
"""


import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import adjusted_rand_score,confusion_matrix, recall_score
import heapq
import warnings
import datetime
import multiprocessing
from scipy import sparse

warnings.filterwarnings("ignore")



def SparseLasso(X,Y,k=5):    
    lassoModel = LassoCV(cv=k).fit(X,Y)
    indi = np.where(lassoModel.alphas_ == lassoModel.alpha_)[0]
    maxcv = lassoModel.mse_path_[indi,:].mean() + lassoModel.mse_path_[indi,:].std()/np.sqrt(k)
    best = np.where(lassoModel.mse_path_.mean(axis=1) < maxcv)[0][0]
    lassoModel = Lasso(alpha = lassoModel.alphas_[best]).fit(X,Y)
    betaFit = lassoModel.coef_
    return betaFit



def relocate(group_update,k):
    group_true_cnt = np.arange(1,k+1)
    group_tmp_cnt =  np.unique(group_update)
    group_lack = list(set(group_true_cnt) - set(group_tmp_cnt))
    gcount = pd.value_counts(group_update)
    if group_lack:
        for i in range(len(group_lack)):
            gcount = gcount.append(pd.Series(0,index = [group_lack[i]]))
    u = np.where(gcount<5)[0]
        
    if len(u):  
        spa = np.array(gcount.index[u])  
        nspa =  np.array(gcount.index[np.where(gcount > 10)[0]])  
        nspa_ind = np.where(np.in1d(group_update,list(nspa)))[0]
        for ispa in range(len(spa)):               
            relocate_ind = np.random.randint(0,len(nspa_ind),size = 5)
            group_update[nspa_ind[relocate_ind]] = spa[ispa]
            nspa_ind = np.delete(nspa_ind,relocate_ind)
    return group_update



def kmeans(X,Y,k,group_init = None):
    features, points = X.shape
    dist = np.zeros((k, points))
    group_update = np.zeros(points)
    switched = True
    ite = 0
    if group_init is None:
        group_init = np.random.randint(1, k+1,size = points)
    
    group_init = relocate(group_init,k)
    
    group = group_init.copy()
    while switched and ite<200:
        group_set = np.unique(group)
        ite +=1  
        #print ("*******current iteration: " + str(ite)+ "**********")
        for i in range(len(group_set)):
            X_tmp = X[:,group == group_set[i]]
            Y_tmp = Y[group == group_set[i]]
#            reg_tmp = LassoCV(cv=5, max_iter=5000,n_alphas = 30).fit(X_tmp.T,Y_tmp)
            if ite > 0:
                reg_tmp = LassoCV(cv=3,n_alphas = 30).fit(X_tmp.T,Y_tmp)
            else:
                reg_tmp = LassoCV(cv=3,alphas = [0.01]).fit(X_tmp.T,Y_tmp)
            beta_tmp = reg_tmp.coef_

            dist[i,:] = (Y - beta_tmp.dot(X))**2    #rows: group; columns: distance from center to point i 1 thru N
        for point in range(points):
            dt = dist[:,point].tolist()
            group_update[point] = dt.index(min(dt)) + 1
            group_update = group_update.astype(int)
        
        group_update = relocate(group_update,k)
        switched = sum(group_update != group)
        group = group_update.copy()

    group_set = np.unique(group_update)
    beta_final = np.zeros((features,len(group_set)))
    for i in range(len(group_set)):
        X_tmp = X[:,group_update == group_set[i]]
        Y_tmp = Y[group_update == group_set[i]]
        beta_final[:,i] = SparseLasso(X_tmp.T,Y_tmp)
    #print("k-means finished, iteration: " + str(ite))    
    return group_update, group_init, beta_final





def swkmeans(X, Y, k, lamb, group_init = None):
    m_features, N_points = X.shape
    dist = np.zeros((k, N_points))
    A = np.zeros((k-1,k))
    for i in range(k-1):
        A[i,i:(i+2)] = np.array([-1,1])
    A0 = A.copy()
    for i in range(N_points-1):        
        A = np.concatenate((A,A0),axis = 1)
    M = np.zeros((N_points,N_points*k))
    for i in range(N_points):
        M[i,k*i:k*(i+1)] = 1     
    center = np.zeros(m_features*k).reshape(m_features,k)

    E_k = np.ones(k-1)
    E_n = np.ones(N_points)

    obj_old = 1e300
    obj_new = 1e200
    ite = 0
    ## initialization
    if group_init is None:
        group_init = np.random.randint(1, k+1,size = N_points)
    group_set = np.unique(group_init)

    for i in range(k):            
        X_tmp = X[:,group_init == group_set[i]]
        Y_tmp = Y[group_init == group_set[i]]
        reg_tmp = LassoCV(cv=3, max_iter=10000).fit(X_tmp.T,Y_tmp)
        beta_tmp = reg_tmp.coef_
        dist[i,:] = (Y - beta_tmp.dot(X))**2   
    D = np.diag(dist.T.reshape(-1))

## update begins
    while (obj_old - obj_new)  > 1e-5 or ite<20:
        ite += 1

        ## update weight       
        ### calculate gamma
        tmp = np.linalg.inv(M.dot(np.linalg.inv(D)).dot(M.T))
        gam = tmp.dot(-2*E_n - M.dot(np.linalg.inv(D)).dot(lamb*A.T.dot(E_k)))
        ### calculate weight
        U = -np.linalg.inv(D).dot(lamb*A.T.dot(E_k)+M.T.dot(gam))/2
        weights = U.reshape(N_points,k).T
        z = list(set(np.where(weights<0)[1]))
        weights[weights < 0] = 0
        weights[:,z] = weights[:,z]/weights[:,z].sum(axis=0)

        if np.min(weights) < 0: #check for underflow
            print("weight vector small")
            break
        for l in range(k):
            w = np.sqrt(weights[l,:])
            Yw = w*Y
            Xw = w*X
            if ite > np.log(np.sqrt(m_features))/np.log(1.06)*1.5:
                center[:,l] = LassoCV(cv=5, max_iter=5000,n_alphas = 30).fit(Xw.T,Yw).coef_
            else:
                center[:,l] = LassoCV(cv=5, max_iter=5000,alphas = [0.01]).fit(Xw.T,Yw).coef_
        for i in range(k):
            dist[i,:] = (Y - center[:,i].dot(X))**2    #rows: group; columns: distance from center to point i 1 thru N
        D = np.diag(dist.T.reshape(-1))
        obj_old = obj_new
        obj_new = (dist*weights**2).sum()
#        print("obj_old: " + str(obj_old) + "; obj_curr: " + str(obj))
        #print("gap"+str(obj_old-obj_new))

    group = np.zeros(N_points)            
    for point in range(N_points):
        dt = dist[:,point].tolist()
        group[point] = dt.index(min(dt)) + 1
    group_set = np.unique(group)
    beta_final = np.zeros((m_features,len(group_set)))
    for i in range(len(group_set)):
        X_tmp = X[:,group == group_set[i]]
        Y_tmp = Y[group == group_set[i]]
        beta_final[:,i] = SparseLasso(X_tmp.T,Y_tmp)
#        beta_final[:,i] = pySCAD(X_tmp.T,Y_tmp)
    print("iteration times:" + str(ite))

    return group, group_init, center, beta_final, weights





def rep_swkmeans(X, Y, k = 2, lamb = 0.1, rep_time = 10):
    tresi = []
    tbic = []
    bic_bst = 10000
    n = X.shape[1]
    starttime = datetime.datetime.now()

    for t in range(rep_time):
        group_p1, group_init, center, beta_final, weight = swkmeans(X, Y, k, lamb, group_init = None)

        group_set = np.unique(group_p1)
        resi2 = np.zeros(n)
        for l in range(len(group_set)):
            X_tmp = X[:,group_p1 == group_set[l]]
            Y_tmp = Y[group_p1 == group_set[l]]
            resi2[group_p1==group_set[l]] = (Y_tmp - beta_final[:,l].dot(X_tmp))**2           
        resi_update = resi2.sum()
        bic_update = criteria_bic(resi_update, beta_final,n)
        tresi.append(resi_update)
        tbic.append(bic_update)

    
    #    if resi_update < resi_bst:
        if bic_update < bic_bst:
            group_bst = group_p1.copy()
            beta_bst = beta_final.copy()
            rpe_bst = np.sqrt(resi_update/n)
            bic_bst = bic_update
            weight_bst =  weight
            center_bst = center.copy()
    
    weight_adj, beta_adj, group_adj = justify(X,Y,weight_bst,group_bst)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).total_seconds()

    print ("Time used:",endtime - starttime)
    ttt = pd.DataFrame(np.vstack((tresi,tbic)).T,columns = ['resi','bic'])
    evalu = pd.DataFrame(np.vstack((rpe_bst,bic_bst,time)).T,columns = ['rpe','bic','time'])

    return group_bst, group_adj, beta_bst, beta_adj, weight_bst, weight_adj, center_bst, evalu, ttt



def swkmeans_app(X, Y, k, lamb, group_init = None):
    m_features, N_points = X.shape
    dist = np.zeros((k, N_points))
    A = np.zeros((k-1,k))
    for i in range(k-1):
        A[i,i:(i+2)] = np.array([-1,1])
    A0 = A.copy()
    for i in range(N_points-1):        
        A = np.concatenate((A,A0),axis = 1)
    M = np.zeros((N_points,N_points*k))
    for i in range(N_points):
        M[i,k*i:k*(i+1)] = 1     
    center = np.zeros(m_features*k).reshape(m_features,k)

    E_k = np.ones(k-1)
    E_n = np.ones(N_points)

    obj_old = 1e300
    obj_new = 1e200
    ite = 0
## initialization
    if group_init is None:
        group_init = np.random.randint(1, k+1,size = N_points)
    group_set = np.unique(group_init)

    for i in range(k):            
        X_tmp = X[:,group_init == group_set[i]]
        Y_tmp = Y[group_init == group_set[i]]
        reg_tmp = LassoCV(cv=3, max_iter=10000).fit(X_tmp.T,Y_tmp)
        #reg_tmp = RidgeCV(cv=5).fit(X_tmp.T,Y_tmp)
        beta_tmp = reg_tmp.coef_
#        beta0 = np.vstack((np.hstack((np.ones(3),np.zeros(m_features-3))),np.hstack((np.ones(3)*-1,np.zeros(m_features-3))))).T
#        beta_tmp = beta_init[:,i]
        dist[i,:] = (Y - beta_tmp.dot(X))**2    #rows: group; columns: distance from center to point i 1 thru N
    D = np.diag(dist.T.reshape(-1))

## update begins
    while (obj_old - obj_new)  > 1e-1 or ite<20:
        #abs((obj_old - obj))  > 1e-5 or /(obj_old*np.sqrt(m_features)))
        ite += 1
        
#
#        U = cp.Variable(n*k)
#        obj = cp.Minimize(cp.square(U).T @ D @ E_kn + 0.1 * A @ U @ E_k.T)
#        constraints = [U<=0.98, M@U-E_n==0]
#        prob = cp.Problem(obj, constraints)
#        prob.solve()
#        weights = (U.value).reshape(N_points,k).T

## update weight       
        ## calculate gamma
        tmp = np.linalg.inv(M.dot(np.linalg.inv(D)).dot(M.T))
        gam = tmp.dot(-2*E_n - M.dot(np.linalg.inv(D)).dot(lamb*A.T.dot(E_k)))
        ## calculate weight
        U = -np.linalg.inv(D).dot(lamb*A.T.dot(E_k)+M.T.dot(gam))/2
        weights = U.reshape(N_points,k).T
        z = list(set(np.where(weights<0)[1]))
        weights[weights < 0] = 0
        weights[:,z] = weights[:,z]/weights[:,z].sum(axis=0)

        if np.min(weights) < 0: #check for underflow
            print("weight vector small")
            break
        for l in range(k):
            w = np.sqrt(weights[l,:])
            Yw = w*Y
            Xw = w*X
            if ite > np.log(np.sqrt(m_features))/np.log(1.06)*1.5:
                center[:,l] = LassoCV(cv=5, max_iter=5000,n_alphas = 30).fit(Xw.T,Yw).coef_
#                center[:,l] = LassoCV(cv=5, max_iter=5000).fit(Xw.T,Yw).coef_
            else:
                center[:,l] = LassoCV(cv=5, max_iter=5000,alphas = [0.01]).fit(Xw.T,Yw).coef_
#                center[:,l] = LassoCV(cv=5, max_iter=5000).fit(Xw.T,Yw).coef_
        for i in range(k):
            dist[i,:] = (Y - center[:,i].dot(X))**2    #rows: group; columns: distance from center to point i 1 thru N
        D = np.diag(dist.T.reshape(-1))

        #dist = np.reshape(np.concatenate(dist, axis=0),(k,N_points))
        #print(dist.shape)
        obj_old = obj_new
        obj_new = (dist*weights**2).sum()
        #print("obj_old: " + str(obj_old) + "; obj_curr: " + str(obj_new))

    group = np.zeros(N_points)            
    for point in range(N_points):
        #dt = dist[:,point].tolist()
        #group[point] = dt.index(min(dt)) + 1
        dt = weights[:,point].tolist()
        group[point] = dt.index(max(dt)) + 1

    group_set = np.unique(group)
    beta_final = np.zeros((m_features,len(group_set)))
    for i in range(len(group_set)):
        X_tmp = X[:,group == group_set[i]]
        Y_tmp = Y[group == group_set[i]]
        beta_final[:,i] = SparseLasso(X_tmp.T,Y_tmp)
#        beta_final[:,i] = pySCAD(X_tmp.T,Y_tmp)
    print("iteration times:" + str(ite))

    return group, group_init, center, beta_final, weights



#####parallel
def rep_kmeans2(X, Y, k, rep_time = 10):
    bic_bst = 10000
    n = X.shape[1]
    tresi = []
    tbic = []
    #tari = []

    group_rep = []
    starttime = datetime.datetime.now()
    for t in range(rep_time):
        #print("current iteration: " + str(t) + "")
        group_p1,group_init, beta_final = kmeans(X,Y,k) 
        group_set = np.unique(group_p1)
        resi2 = np.zeros(n)
        for l in range(len(group_set)):
            X_tmp = X[:,group_p1 == group_set[l]]
            Y_tmp = Y[group_p1 == group_set[l]]
            resi2[group_p1==group_set[l]] = (Y_tmp - beta_final[:,l].dot(X_tmp))**2           
        resi_update = resi2.sum()
        bic_update = criteria_bic(resi_update, beta_final,n)
        group_rep.append(group_p1)
        tresi.append(resi_update)
        tbic.append(bic_update)

    #    if resi_update < resi_bst:
        if bic_update < bic_bst:
            group_bst = group_p1.copy()
            beta_bst = beta_final.copy()
            rpe_bst = np.sqrt(resi_update/n)
            bic_bst = bic_update
     
    endtime = datetime.datetime.now()
    time = (endtime - starttime).total_seconds()
    #print ("Time used:",endtime - starttime)
    evalu = pd.DataFrame(np.vstack((rpe_bst,bic_bst,time)).T,columns = ['rpe','bic','time'])
    ttt = pd.DataFrame(np.vstack((tresi,tbic)).T,columns = ['resi','bic'])

    return group_bst, beta_bst, evalu, ttt, group_rep



def par_kmeans(X, Y):
    group_bst, beta_bst, evalu, ttt, group_rep = rep_kmeans2(X,Y, 2, rep_time = 5)
    resi_i = min(ttt['resi'])
    return resi_i, beta_bst


def par(X,Y,seq,core=2):
    cores = core
    #cores = multiprocessing.cpu_count()
    multi_pool = multiprocessing.Pool(processes=cores)
    result = []
    par_sequence = []
    for i in range(seq.shape[0]):
        #print("********current screening:" + str(i))
        result.append(multi_pool.apply_async(par_kmeans,(X[seq[i],:],Y)))
        par_sequence.append(seq[i])
    multi_pool.close()
    multi_pool.join()
    
    resi_pool = []
    beta_pool = []
    for i in range(seq.shape[0]):
        res = result[i].get()
        resi_pool.append(res[0])
        beta_pool.append(res[1])
    
    return par_sequence, resi_pool, beta_pool


def par_scr(X,Y,m1 = 5,m2 = 5,core = 2):
    p = X.shape[0]
    n = X.shape[1]
        
    seq = np.random.permutation(np.arange(0,p))
    seq = seq.reshape(-1,m1)
    
    ####并行
    starttime = datetime.datetime.now()
    par_sequence, resi_pool, beta_pool = par(X,Y,seq,core)
    endtime = datetime.datetime.now()
    print ("Time used:",endtime - starttime)

    resi_sig = np.hstack((np.array(resi_pool).reshape(-1,1),par_sequence))   
    
    sig_indx = list(map(resi_pool.index,heapq.nsmallest(m2, resi_pool))) 
    
    
    par_sequence = np.array(par_sequence)
    sig_gro_p = par_sequence[sig_indx,:]    
    sig_grp_beta = []              
    for i in range(len(sig_indx)):
        sig_grp_beta.append(beta_pool[sig_indx[i]])
    
    sig_beta_sum = np.array([np.sum(abs(c),axis=1) for c in sig_grp_beta]) 
    
    sig_p = sig_gro_p[sig_beta_sum != 0]
    #sig_p2 = sig_gro_p[sig_beta_sum >0.05]

    time = (endtime - starttime).total_seconds()
    
    return sig_p, time, resi_sig



def justify(X,Y,weights,group,beta_final):
    m_features,N = X.shape
    group_set = np.unique(group)
    ind = np.array(())
    group_up = group.copy()
    weight_up = weights.copy()

    for j in range(len(group_set)):        
        tmp = np.where(weights[j,:]>0.93)[0]
        if len(tmp)>5:
            ind = np.append(ind,tmp)
    ind = ind.astype(int)
    
    if len(ind): 
        X0 = X[:,ind]
        Y0 = Y[ind]
        group_wk0  = group[ind]
        group_set0 = np.unique(group_wk0)
        for i in range(len(np.unique(group_wk0))):        
            X_tmp = X0[:,group_wk0 == group_set0[i]]
            Y_tmp = Y0[group_wk0 == group_set0[i]]
            beta_final[:,int(group_set0[i]-1)] = SparseLasso(X_tmp.T,Y_tmp)
        
        X1 = np.delete(X,ind,axis=1)
        Y1 = np.delete(Y,ind,axis=0)
        
        est = beta_final[:,0].dot(X1)
        for i in range(1,len(group_set)):
            est = np.vstack((est,beta_final[:,i].dot(X1)))
        
        est_er = 1/abs(est-Y1)
        
        est_w = est_er/sum(est_er)
        
        ind2 = np.delete(np.arange(0,N),ind,axis=0)
        weight_up[:,ind2] = est_w
        group_up = np.zeros(N)
        for i in range(N):
            group_up[i] = np.argmax(weight_up[:,i]) + 1
    return weight_up, beta_final, group_up
    


def criteria_bic(sse, beta_hat,n):
    beta_sig = np.sign(beta_hat**2).sum()
    bic = n*np.log(sse/n)+0.7*beta_sig*np.log(n)
    return bic


def k_bic(sse,k,sig_p,n):
    bic = n*np.log(sse/n)+np.log(n)*(2*k-1+sig_p)
    return bic



def norm2(x):
    return np.sqrt((x**2).sum())
