%======================================================================
% File:        gs.m
% Description: solve e.g. 2.3.1 using golden section
% 
% Created on:  Tue Sep 23 2025 09:52:32
% Author:      Shen Yang
% University:  Hunan Normal University
% Email:       yangshen@hunnu.edu.cn
%======================================================================

a = 0;
b = 3;

phi = @(x) 3*x^4 - 16*x^3 + 30*x^2 - 24*x + 8;

golden_section = (sqrt(5) - 1) / 2;

u = b - golden_section * (b - a);
v = a + golden_section * (b - a);
phiu = phi(u);
phiv = phi(v);

iter = 0;
max_iter = 1000;

tol = 1e-12;

while (b - a) > tol && iter < max_iter

    iter = iter + 1;
    if abs(phiu - phiv) < tol + tol * max([1, abs(phiu), abs(phiv)])
        a = u;
        b = v;
        u = b - golden_section * (b - a);
        v = a + golden_section * (b - a);
        phiu = phi(u);
        phiv = phi(v);
    elseif phiu < phiv
        b = v;
        v = u;
        u = b - golden_section * (b - a);
        phiv = phiu;
        phiu = phi(u);
    else % phiu > phiv
        a = u;
        u = v;
        v = a + golden_section * (b - a);
        phiu = phiv;
        phiv = phi(v);
    end
end

x = (a + b) / 2;
phix = phi(x);
fprintf("%.4f\n",x);
fprintf("%.4f",phix);