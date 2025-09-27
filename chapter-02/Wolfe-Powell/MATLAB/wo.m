%======================================================================
% File:        wo.m
% Description: solve e.g. 2.3.1 using Wolfe-Powell rule
% 
% Created on:  Tue Sep 23 2025 11:03:24
% Author:      Shen Yang
% University:  Hunan Normal University
% Email:       yangshen@hunnu.edu.cn
%======================================================================

a = 0; b = 3;
f = @(x) 3*x.^4 - 16*x.^3 + 30*x.^2 - 24*x + 8;
grad_f = @(x) 12*x.^3 - 48*x.^2 + 60*x - 24;

x = (a + b)/2;
tol = 1e-8; 

beta   = 1.0;   
rho    = 0.5;   
rho1   = 0.5;
sigma1 = 0.1; 
sigma2 = 0.9;

iter = 0;
max_iter = 1000;

while abs(grad_f(x)) > tol && iter < max_iter
    iter = iter + 1;
    d  = -grad_f(x);                
    aL = wolfe_powell(x, d, f, grad_f, beta, rho, sigma1, rho1, sigma2);
    x  = x + aL * d;
end

fprintf("iter: %d\n",iter);
fprintf("x: %.4f\n",x);
fprintf("f(x): %.4f",f(x));

function alpha = wolfe_powell(x, d, f, grad_f, beta, rho, sigma1, rho1, sigma2)

    fx = f(x);
    gx = grad_f(x);
    phi_prime0 = gx * d;

    armijo_ok = @(a) f(x + a*d) <= fx + sigma1 * a * phi_prime0;
    wolfe_ok = @(a) (grad_f(x + a*d) * d) >= sigma2 * phi_prime0;  

    % step 0
    if armijo_ok(1) && wolfe_ok(1)
        alpha = 1; 
        return;
    end

    % step 1: 
    % find alpha0 in {beta*rho^i | i=0,±1,±2,...}, beta_up = rho^{-1} * alpha0
    maxI = 200;  
    
    a_i = beta * rho^0;
    ok0 = armijo_ok(a_i);

    if ok0
        % i = -1,-2,... until first failure
        i_best = 0;
        for step = 1:maxI
            i_try = -step;
            a_try = beta * rho^i_try;  
            if armijo_ok(a_try)
                i_best = i_try;
            else
                break;  
            end
        end
        alpha0 = beta * rho^i_best;    
    else
        % i=0 failed, i = 1,2,... until first success
        i_hit = NaN;
        for step = 1:maxI
            i_try = step;
            a_try = beta * rho^i_try;
            if armijo_ok(a_try)
                i_hit = i_try; 
                break;
            end
        end
        alpha0 = beta * rho^i_hit;  % i_hit is the biggest i with success
    end

    % beta_up = rho^{-1} * alpha0
    beta_up = alpha0 / rho;

    % step 2 & 3
    alpha_i = alpha0;
    guard = 0; 
    guardMax = 500;

    while guard < guardMax
        % step 2
        if wolfe_ok(alpha_i)
            alpha = alpha_i; 
            return;
        end

        % step 3
        % find the largest alpha in {alpha_i + rho1^j (beta_up - alpha_i)} that satisfies Armijo
        current = beta_up;
        if ~armijo_ok(current)
            while ~armijo_ok(current) && guard < guardMax
                current = alpha_i + rho1 * (current - alpha_i);  % current = alpha_i + rho1^j (beta_up - alpha_i)
                guard = guard + 1;
                if (current - alpha_i) < 1e-16
                    break; 
                end
            end
        end

        % the largest one becomes alpha_{i+1}
        alpha_i = current;
        beta_up = alpha_i / rho;    
        guard = guard + 1;
    end

    alpha = alpha_i;
end

