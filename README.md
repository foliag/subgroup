# Regression-based heterogeneity analysis to identify overlapping subgroup structure in high-dimensional data
This is a `Python` implementation of the following paper:
Luo, Z., Yao, X., Sun, Y., Fan, X. (2022). Regression-based heterogeneity analysis to identify overlapping subgroup structure in high-dimensional data.

# Introduction
This algorithm uses an alternating optimization to obtain the partial minimum of the objective function:
<a href="https://www.codecogs.com/eqnedit.php?latex=\min_{A,&space;U}\&space;\&space;\sum_{i=1}^{n}\sum_{k=1}^{K}u_{ki}^m(y_i&space;-&space;X_i&space;\alpha_k)^2&space;&plus;&space;\sum_{k=1}^K\lambda_k&space;\|{\alpha}_k\|_1&space;&plus;&space;\gamma&space;\sum_{i=1}^n\sum_{k=2}^K(u_{(k),i}-u_{(k-1),i})^2&space;\\&space;\text{s.t.}\&space;\&space;\sum_{k=1}^K&space;u_{ki}&space;=&space;1,&space;\&space;0&space;\leq&space;u_{ki}&space;\leq&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_{A,&space;U}\&space;\&space;\sum_{i=1}^{n}\sum_{k=1}^{K}u_{ki}^m(y_i&space;-&space;X_i&space;\alpha_k)^2&space;&plus;&space;\sum_{k=1}^K\lambda_k&space;\|{\alpha}_k\|_1&space;&plus;&space;\gamma&space;\sum_{i=1}^n\sum_{k=2}^K(u_{(k),i}-u_{(k-1),i})^2&space;\\&space;\text{s.t.}\&space;\&space;\sum_{k=1}^K&space;u_{ki}&space;=&space;1,&space;\&space;0&space;\leq&space;u_{ki}&space;\leq&space;1" title="\min_{A, U}\ \ \sum_{i=1}^{n}\sum_{k=1}^{K}u_{ki}^m(y_i - X_i \alpha_k)^2 + \sum_{k=1}^K\lambda_k \|{\alpha}_k\|_1 + \gamma \sum_{i=1}^n\sum_{k=2}^K(u_{(k),i}-u_{(k-1),i})^2 \\ \text{s.t.}\ \ \sum_{k=1}^K u_{ki} = 1, \ 0 \leq u_{ki} \leq 1" /></a>

The algorithm starts from an initial estimate of U, and then updates A and U sequentially until the convergence is reached.

# Requirements
* Python3
* Package numpy; panads; sklearn.linear_model; sklearn.metrics; scipy;

# Contents
* All usable functions:
   `par_scr`, `rep_kmeans2`, `rep_swkmeans`, `kmeans`, `swkmeans`, `swkmeans_app`, `justify`, `bic_run`, `gen_beta_nonoverlap`, `gen_var_nonoverlap`, `gen_beta_overlap`, `gen_var_overlap`, `sse_calculate_hard`, `sse_calculate_soft`, `rmse_multi_hard`, `rmse_multi_soft`

# Demo
* Generate simulated data under `s1` setting
```
import heteroverlap as ho

p=100   #scaled variable dimension to 100
n = 200
rho=0.5
e = 0.5
k=2
prop = np.array((0.4,0.4,0.2))    #proportion of sample size in each group
a = np.array([[1,2,3,0,0,0],[0,0,0,-4,-5,-6]])   #non-zero coefficients
npro = int(n*prop[-1])
pp_tr = 6  #total number of non-zero variables
np_tr = p-pp_tr   #total number of insignificant variables

beta,weight_overlap = ho.gen_beta_overlap(p,a,npro,k)
Y,X,group_true, err = ho.gen_var_overlap(n,p,prop,k,e,beta)


```
* Implement algorithm through three steps 
```

## screening
print("%%%%%screening begins%%%%%")
sig_p,time, resi_sig = ho.par_scr(X,Y,m1 = 5, m2 = 5,core = 2)

## inital estimate
print("%%%%%inital estimate begins%%%%%")
X_sig = X[sig_p,:]
group_k2, beta_k, evalu_k, ttt_k, group_rep_k = ho.rep_kmeans2(X_sig, Y, k=2, rep_time = 10)

## final estimate 
print("%%%%%final estimate begins%%%%%")
group_est, group_init, center, beta_est, weight_est = ho.swkmeans(X, Y, k, lamb=0, group_init = group_k2)
weight_up, beta_up, group_up = ho.justify(X,Y,weight_est,group_est,beta_est)
```
* Evaluation
```
rpe = np.sqrt(ho.sse_calculate_soft(beta_est,weight_est,X,Y)/n)
rmse = ho.rmse_multi_soft(beta, beta_up, weight_up,n,k,prop,p)
# users can add more evaluation metrics like ARI and L1 loss. 
```
