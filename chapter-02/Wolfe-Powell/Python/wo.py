# -*- coding: utf-8 -*-
"""
File:        wo.py
Description: 

Created on:  Thu Sep 25 2025 14:59:13
Author:      Shen Yang
University:  Hunan Normal University
Email:       yangshen@hunnu.edu.cn
"""

import numpy as np

def f(x):
    x = np.asarray(x)
    return 3*x**4 - 16*x**3 + 30*x**2 - 24*x + 8

def df(x):
    x = np.asarray(x)
    return 12*x**3 - 48*x**2 + 60*x - 24

def wolfe_powell(x,d,beta,rho,rho1,sigma1,sigma2,f,df):
    
    fx = f(x)
    sigma1_df_d = sigma1 * df(x) * d
    sigma2_df_d = sigma2 * df(x) * d
    
    # step 0
    if f(x + d) <= fx + sigma1_df_d and df(x + d) * d >= sigma2_df_d:
        return 1.0
    
    # step 1
    alpha = beta
    if f(x + alpha * d) > fx + alpha * sigma1_df_d:
        while f(x + alpha * d) > fx + alpha * sigma1_df_d:
            alpha = rho * alpha
    else:
        while f(x + alpha * d) <= fx + alpha * sigma1_df_d:
            alpha = alpha / rho
        alpha = rho * alpha
            
    # step 2, step 3
    while df(x + alpha * d) * d < sigma2_df_d:
        beta_k = alpha / rho
        temp = beta_k
        while f(x + temp * d) > fx + temp * sigma1_df_d:
            temp = (temp - alpha) * rho1 + alpha
            if temp - alpha < 1e-16:
                return alpha
        alpha = temp
        
    return alpha

a = 0
b = 3
x = (a + b) / 2

tol = 1e-8
iter = 0
iter_max = 1000

beta = 1
rho = 0.5
rho1 = 0.5
sigma1 = 0.1
sigma2 = 0.9

while abs(df(x)) > tol and iter < iter_max:
    iter += 1
    d = -df(x)
    alpha = wolfe_powell(x,d,beta,rho,rho1,sigma1,sigma2,f,df)
    x = x + alpha * d
    
print("x: ", x)
print("f(x): ", f(x))
