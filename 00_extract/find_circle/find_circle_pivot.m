%% Smallest Enclosing Circle in 2D (Pivot-style algorithm, pure-threshold stop)
%  - Pivot high-level loop: in each iteration choose the farthest outside point
%    (maximum excess), force it onto the boundary, and solve a constrained
%    sub-problem (Welzl-style).
%  - Uses distance-squared comparisons + mixed tolerance (relative+absolute).
%  - No iteration cap: termination relies solely on the threshold condition.
%
%  Input:
%    P: n-by-2 matrix, each row is [x, y]
%  Output:
%    O: 1-by-2 center
%    R: radius (scalar)

function [O, R] = find_circle_pivot(P)
    n = size(P, 1);

    % ---------- trivial cases ----------
    if n == 0
        O = [0, 0]; R = 0; return;
    elseif n == 1
        O = P(1, :); R = 0; return;
    elseif n == 2
        O = (P(1, :) + P(2, :)) / 2;
        R = 0.5 * norm(P(1, :) - P(2, :));
        return;
    end

    % ---------- randomize order (improves expected runtime) ----------
    P = P(randperm(n), :);

    % Mixed tolerance
    rel_eps = 1e-12;
    abs_eps = 1e-24;

    % ---------- initialize with first two points ----------
    [O, R2] = circle_from_2pts(P(1, :), P(2, :));

    % ---------- pivot main loop (pure threshold termination) ----------
    while true
        % compute all distances squared and excesses
        d2_all = sum((P - O).^2, 2);
        excess_all = d2_all - R2;

        % find point with maximum excess
        [~, k] = max(excess_all);

        % threshold (same rule as is_outside)
        threshold = R2 * (1 + rel_eps) + abs_eps;

        % termination condition
        if d2_all(k) <= threshold
            break;
        end

        % pivot step: enforce P(k,:) on boundary
        [O, R2] = min_circle_with_point(P, P(k, :), rel_eps, abs_eps);

        % move pivot to front
        if k ~= 1
            P = [P(k,:); P(1:k-1,:); P(k+1:end,:)];
        end
    end

    R = sqrt(R2);
end

%% Constrained: smallest circle that contains "points", with boundary point "p"
%  Loop from 1..m:
%   - i==1: initialize circle from (p, points(1,:))
%   - i>=2: if points(i,:) is outside, recurse to two-fixed-boundary case
function [O, R2] = min_circle_with_point(points, p, rel_eps, abs_eps)
    m = size(points, 1);

    % initialize with (p, points(1,:))
    [O, R2] = circle_from_2pts(p, points(1, :));

    % grow if outside
    for i = 2:m
        if is_outside(points(i, :), O, R2, rel_eps, abs_eps)
            [O, R2] = min_circle_with_two_points(points(1:i-1, :), p, points(i, :), rel_eps, abs_eps);
        end
    end
end

%% Constrained: smallest circle with fixed boundary points p1, p2
%  Loop 1..k; start from diameter circle of (p1,p2); if a point lies outside,
%  switch to the circumcircle of (p1,p2,that_point).
function [O, R2] = min_circle_with_two_points(points, p1, p2, rel_eps, abs_eps)
    k = size(points, 1);

    % initialize with diameter circle of (p1,p2)
    [O, R2] = circle_from_2pts(p1, p2);

    % check all remaining points
    for i = 1:k
        if is_outside(points(i, :), O, R2, rel_eps, abs_eps)
            [O, R2] = circle_from_3pts(p1, p2, points(i, :));
        end
    end
end

%% Helper: circle from two points (diameter circle), return center & radius^2
function [O, R2] = circle_from_2pts(a, b)
    O  = (a + b) / 2;
    v  = a - O;
    R2 = v(1)*v(1) + v(2)*v(2);
end

%% Helper: circle from three points (circumcircle or farthest-pair fallback), return radius^2
function [O, R2] = circle_from_3pts(p1, p2, p3)
    A = p2 - p1; B = p3 - p1;
    area2 = abs(A(1)*B(2) - A(2)*B(1));     % twice triangle area
    scale2 = max([dist2(p1,p2), dist2(p1,p3), dist2(p2,p3), 1]);  % scale guard
    collinear_eps = 1e-14 * (scale2 + 1);

    % nearly collinear → farthest-pair diameter circle
    if area2 < collinear_eps
        d12 = dist2(p1, p2); d13 = dist2(p1, p3); d23 = dist2(p2, p3);
        if d12 >= d13 && d12 >= d23
            [O, R2] = circle_from_2pts(p1, p2);
        elseif d13 >= d23
            [O, R2] = circle_from_2pts(p1, p3);
        else
            [O, R2] = circle_from_2pts(p2, p3);
        end
        return;
    end

    % general circumcircle (analytic)
    E = A(1)*(p1(1)+p2(1)) + A(2)*(p1(2)+p2(2));
    F = B(1)*(p1(1)+p3(1)) + B(2)*(p1(2)+p3(2));
    G = 2 * (A(1)*B(2) - A(2)*B(1));

    % guard for tiny G
    if abs(G) < 1e-18 * (scale2 + 1)
        d12 = dist2(p1, p2); d13 = dist2(p1, p3); d23 = dist2(p2, p3);
        if d12 >= d13 && d12 >= d23
            [O, R2] = circle_from_2pts(p1, p2);
        elseif d13 >= d23
            [O, R2] = circle_from_2pts(p1, p3);
        else
            [O, R2] = circle_from_2pts(p2, p3);
        end
        return;
    end

    Ox = (B(2)*E - A(2)*F) / G;
    Oy = (A(1)*F - B(1)*E) / G;
    O  = [Ox, Oy];
    R2 = dist2(p1, O);
end

%% Utility: squared distance
function d2 = dist2(a, b)
    d = a - b; d2 = d(1)*d(1) + d(2)*d(2);
end

%% Utility: check if point is outside circle (distance-squared + mixed tolerance)
function out = is_outside(p, O, R2, rel_eps, abs_eps)
    d2  = dist2(p, O);
    thr = R2 * (1 + rel_eps) + abs_eps;
    out = (d2 > thr);
end