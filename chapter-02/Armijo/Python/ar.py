# -*- coding: utf-8 -*-
"""
File:        ar.py
Description: 

Created on:  Thu Sep 25 2025 14:29:58
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

def armijo(x,d,beta,rho,sigma1,f,df):
    
    fx = f(x)
    sigma1_df_d = sigma1 * df(x) * d
    
    if f(x + d) <= fx + sigma1_df_d:
        return 1.0
    
    alpha = beta
    while f(x + alpha * d) > fx + alpha * sigma1_df_d:
        alpha = rho * alpha
        
    return alpha

a = 0
b = 3
x = (a + b) / 2

tol = 1e-12
iter = 0
iter_max = 1000

beta = 0.5
rho = 0.5
sigma1 = 0.1

while abs(df(x)) > tol and iter < iter_max:
    iter += 1
    d = -df(x)
    alpha = armijo(x,d,beta,rho,sigma1,f,df)
    x = x + alpha * d
    
print("x: ", x)
print("f(x): ", f(x))
    