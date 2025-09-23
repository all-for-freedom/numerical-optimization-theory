# -*- coding: utf-8 -*-
"""
File:        gs.py
Description: 

Created on:  Tue Sep 23 2025 18:00:09
Author:      Shen Yang
University:  Hunan Normal University
Email:       yangshen@hunnu.edu.cn
"""

import numpy as np

def f(x):
    x = np.asarray(x)
    return 3*x**4 - 16*x**3 + 30*x**2 - 24*x + 8

a = 0
b = 3

golden_ratio = (np.sqrt(5) - 1) / 2

u = b - golden_ratio * (b - a)
v = a + golden_ratio * (b - a)

fu = f(u)
fv = f(v)

seps = 1e-6
atol = 1e-12
rtol = 1e-12

iter = 0
max_iter = 1000

while (b - a) > seps and iter < max_iter:
    
    iter += 1
    
    if abs(fu - fv) < atol + rtol * max(1, abs(fu), abs(fv)):
        a = u
        b = v
        u = b - golden_ratio * (b - a)
        v = a + golden_ratio * (b - a)
        fu = f(u)
        fv = f(v)
    elif fu < fv:
        b = v
        v = u
        u = b - golden_ratio * (b - a)
        fv = fu
        fu = f(u)
    else:
        a = u
        u = v
        v = a + golden_ratio * (b - a)
        fu = fv
        fv = f(v)

x = (a + b) / 2

print(f"x = {x}")
print(f"f(x) = {f(x)}")
