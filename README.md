# Regression-based heterogeneity analysis to identify overlapping subgroup structure in high-dimensional data
This is a `Python` implementation of the following paper:
Luo, Z., Yao, X., Sun, Y., Fan, X. (2022). Regression-based heterogeneity analysis to identify overlapping subgroup structure in high-dimensional data.

# Introduction
This algorithm uses an alternating optimization to obtain the partial minimum of the objective function:
<a href="https://www.codecogs.com/eqnedit.php?latex=\min_{A,&space;U}\&space;\&space;\sum_{i=1}^{n}\sum_{k=1}^{K}u_{ki}^m(y_i&space;-&space;X_i&space;\alpha_k)^2&space;&plus;&space;\sum_{k=1}^K\lambda_k&space;\|{\alpha}_k\|_1&space;&plus;&space;\gamma&space;\sum_{i=1}^n\sum_{k=2}^K(u_{(k),i}-u_{(k-1),i})^2&space;\\&space;\text{s.t.}\&space;\&space;\sum_{k=1}^K&space;u_{ki}&space;=&space;1,&space;\&space;0&space;\leq&space;u_{ki}&space;\leq&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_{A,&space;U}\&space;\&space;\sum_{i=1}^{n}\sum_{k=1}^{K}u_{ki}^m(y_i&space;-&space;X_i&space;\alpha_k)^2&space;&plus;&space;\sum_{k=1}^K\lambda_k&space;\|{\alpha}_k\|_1&space;&plus;&space;\gamma&space;\sum_{i=1}^n\sum_{k=2}^K(u_{(k),i}-u_{(k-1),i})^2&space;\\&space;\text{s.t.}\&space;\&space;\sum_{k=1}^K&space;u_{ki}&space;=&space;1,&space;\&space;0&space;\leq&space;u_{ki}&space;\leq&space;1" title="\min_{A, U}\ \ \sum_{i=1}^{n}\sum_{k=1}^{K}u_{ki}^m(y_i - X_i \alpha_k)^2 + \sum_{k=1}^K\lambda_k \|{\alpha}_k\|_1 + \gamma \sum_{i=1}^n\sum_{k=2}^K(u_{(k),i}-u_{(k-1),i})^2 \\ \text{s.t.}\ \ \sum_{k=1}^K u_{ki} = 1, \ 0 \leq u_{ki} \leq 1" /></a>

The algorithm starts from an initial estimate of U, and then updates A and U sequentially until the convergence is reached.

# Requirements
* Python3
* Package numpy; panads; math; sklearn.linear_model; sklearn.metrics; scipy; heapq; datetime; multiprocessing

# Contents
* `src.py`: main function to run our algorithm, see demo below.
* `utils.py`: code for simulated data generation and results evaluation.

# Demo
* Generate simulated data under `s1` setting
```
import src
import utils

p=1000
n = 200
rho=0.5
e = 0.5
k=2
prop = np.array((0.4,0.4,0.2))    #proportion of sample size in each group
a = np.array([[1,2,3,0,0,0],[0,0,0,-4,-5,-6]])   #non-zero coefficients
npro = int(n*prop[-1])
pp_tr = 6  #positive p
np_tr = p-pp   #negative p

beta,weight_overlap = gen_beta_overlap(p,a,npro,k)
Y,X,group_true, err = gen_var_overlap(n,p,rho,prop,k,e)

```
* Implement algorithm through three steps 
```
##initial estimate
core = multiprocessing.cpu_count()
sig_p, time, resi_sig = src.par_scr(X,Y,m1 = 5, m2 = 5,core = 3)
X_sig = X[sig_p,:]
group_k2, beta_k, evalu_k, ttt_k, group_rep_k = src.rep_kmeans2(X_sig, Y, k, rep_time = 5)

##estimate A and U
group_est, group_init, center, beta_est, weight_est = src.swkmeans(X, Y, k, lamb=0.1, group_init = group_k2)

##final adjustment (optional)
weight_up, beta_up, group_up = src.justify(X,Y,weight_est,group_est,beta_est)
```
* evaluation
```
rpe = np.sqrt(sse_calculate(beta_est,weight_est,X,Y)/n)
ari = adjusted_rand_score(group_true, group_up)         # for s4-s5
l1loss = l1_loss(group_true,weight_overlap,weight_up)   # for s1-s3
rmse = rmse_multi(beta, beta_up, weight_up,n)
fp,tp = confu(pp_tr,beta_up)
```

