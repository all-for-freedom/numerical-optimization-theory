%======================================================================
% File:        ar.m
% Description: solve e.g. 2.3.1 using Armijo rule, to use armijo rule, 
%              we need to provide a descent direction, here we use -grad(phi)
% 
% Created on:  Tue Sep 23 2025 10:04:15
% Author:      Shen Yang
% University:  Hunan Normal University
% Email:       yangshen@hunnu.edu.cn
%======================================================================

a = 0; 
b = 3;
f = @(x) 3*x.^4 - 16*x.^3 + 30*x.^2 - 24*x + 8;
grad_f = @(x) 12*x.^3 - 48*x.^2 + 60*x - 24;

x = (a + b)/2;
tol = 1e-8; 

beta = 0.5;   
rho = 0.5;   
sigma1 = 0.1; 

iter = 0;
max_iter = 1000;

while abs(grad_f(x)) > tol && iter < max_iter
    iter = iter + 1;
    d  = -grad_f(x);                
    aL = armijo(x, d, f, grad_f, beta, rho, sigma1);
    x  = x + aL * d;
end

fprintf("iter: %d\n",iter);
fprintf("x: %.4f\n",x);
fprintf("f(x): %.4f",f(x));

function alpha = armijo(x, d, f, grad_f, beta, rho, sigma1)
    
    fx = f(x);
    g = grad_f(x);
    grad_f_d = g * d;     

    if f(x + 1.0 * d) <= fx + sigma1 * 1.0 * grad_f_d
        alpha = 1.0;
        return
    end
    
    alpha = beta;
    while f(x + alpha * d) > fx + sigma1 * alpha * grad_f_d
        alpha = alpha * rho;        
        if alpha < 1e-16           
            break;
        end
    end
end