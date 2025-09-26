# -*- coding: utf-8 -*-
"""
File:        find_way.py
Description: Fast Marching for K(x,y) = inf_{gamma} ∫ (1/d0) ds  on a binary grid

Created on:  Thu Sep 25 2025 17:07:12
Author:      Shen Yang
University:  Hunan Normal University
Email:       yangshen@hunnu.edu.cn
"""

import heapq
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# 1) Load map and build binary grid
# ----------------------------
# map.png：white=1 free，black=0 obstacle
# img = Image.open("./case1/map1.png").convert("L")
# img = Image.open("./case2/map2.png").convert("L")
img = Image.open("./case3/map3.png").convert("L")
arr = np.array(img)
grid = (arr > 128).astype(np.uint8)   # 1=free, 0=obstacle
H, W = grid.shape

# (col, row)）  map 1
# col_row_start = (1750, 200)   # (x=1750, y=200)
# col_row_goal  = (3100, 1400)  # (x=3100, y=1400)

# (703, 363) to (780,664)   map 2 down
# col_row_start = (703, 363)   
# col_row_goal  = (780, 664)   

# (703, 363) to (121,537)   map 2 left
# col_row_start = (703, 363)   
# col_row_goal  = (121, 357)  

# (703, 363) to (1280,328)   map 2 right
# col_row_start = (703, 363)   
# col_row_goal  = (1280, 328)

# (703, 363) to (680,42)   map 2 up
# col_row_start = (703, 363)   
# col_row_goal  = (680, 42)

# (735, 601) to (85, 850)  map 3
col_row_start = (735, 601)   
col_row_goal  = (85, 850)

# convert to (row, col)   
start = (col_row_start[1], col_row_start[0])  # (row, col)
goal  = (col_row_goal[1],  col_row_goal[0])   # (row, col)

# safety check
def in_bounds(rc):
    r, c = rc
    return 0 <= r < H and 0 <= c < W

assert in_bounds(start) and in_bounds(goal), "Start/goal out of bounds"
assert grid[start] == 1 and grid[goal] == 1, "Start/goal must in free area(=1)"

# ----------------------------
# 2) Build distance field d0 and speed F = d0 + eps
# ----------------------------
# To ensure the outer boundary is also treated as an obstacle, we pad the free-space image with a border of zeros.
import numpy as np

def edt_1d(f):
    """
    精确 1D 欧氏距离变换（平方距离），Felzenszwalb & Huttenlocher 算法。
    输入:
        f: 长度 n 的 1D 数组，f[i]=0 表示"源/障碍"，f[i]=+inf 表示其他。
    输出:
        d2: 长度 n 的数组，d2[i] = min_j ( (i-j)^2 + f[j] )
    复杂度: O(n)
    """
    n = f.shape[0]
    v = np.zeros(n, dtype=int)         # 抛物线的索引
    z = np.zeros(n+1, dtype=float)     # 抛物线交点位置
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = +np.inf

    # 前向：构建下包络
    for q in range(1, n):
        # 计算新抛物线与当前包络最后一条抛物线的交点
        while True:
            p = v[k]
            # 求解两抛物线 (x-p)^2+f[p] 与 (x-q)^2+f[q] 的交点 s
            # 展开得: s = ( (f[q] + q^2) - (f[p] + p^2) ) / (2(q - p))
            s = ((f[q] + q*q) - (f[p] + p*p)) / (2.0*(q - p))
            if s > z[k]:
                k += 1
                v[k] = q
                z[k] = s
                z[k+1] = +np.inf
                break
            else:
                k -= 1
                if k < 0:
                    k = 0
                    v[0] = q
                    z[0] = -np.inf
                    z[1] = +np.inf
                    break

    # 反向：根据下包络生成最小值
    d2 = np.empty(n, dtype=float)
    k = 0
    for i in range(n):
        # 找到包含 i 的那条抛物线
        while z[k+1] < i:
            k += 1
        j = v[k]
        d2[i] = (i - j)*(i - j) + f[j]
    return d2

def edt_from_scratch(binary_free):
    """
    2D 精确欧氏距离变换: 到最近障碍(=0)的距离。
    输入:
        binary_free: (H,W) uint8/bool, 1=free, 0=obstacle
    输出:
        d0: (H,W) float64，每个像素到最近 0 的欧氏距离；障碍处为 0。
    """
    H, W = binary_free.shape
    INF = 1e15  # 足够大的数做 +inf

    # 第一步：构造 f0 = 0 在障碍处；+inf 在非障碍处
    f0 = np.where(binary_free == 0, 0.0, INF)

    # 第二步：先按列做 1D 变换（得到各列的平方距离）
    d2_col = np.empty_like(f0, dtype=float)
    for x in range(W):
        d2_col[:, x] = edt_1d(f0[:, x])

    # 第三步：再按行对 d2_col 做 1D 变换（得到最终平方距离）
    d2 = np.empty_like(d2_col, dtype=float)
    for y in range(H):
        d2[y, :] = edt_1d(d2_col[y, :])

    # 第四步：开方得到欧氏距离；障碍处仍为 0
    d0 = np.sqrt(d2, dtype=float)
    return d0

free = (grid == 1).astype(np.uint8)
padded = np.pad(free, 1, mode="constant", constant_values=0)

# Euclidean Distance Transform
# 输出：与 padded 同形状的浮点数组，每个像素值是：
# “到最近一个 0 像素的欧式距离（单位是像素）”
dpad = edt_from_scratch(padded)
d0 = dpad[1:-1, 1:-1].astype(np.float64)

eps = 1.0  # 数值稳定用的小正数（可根据像素尺度调）
F = np.zeros_like(d0, dtype=np.float64)
F[free == 1] = d0[free == 1] + eps  # 障碍保持0速度（不可达）

# ----------------------------
# 3) Fast Marching Method (4-邻域的 Godunov 更新)
# ----------------------------
INF = np.inf
KNOWN, TRIAL, FAR = 2, 1, 0

# T[r,c]：Eikonal 方程 |\nabla T| = 1/F 的数值解，物理上是“从 (r,c) 到终点”最小旅行时间/代价。单位 ≈ 距离 / 速度（像素）。
T = np.full((H, W), INF, dtype=np.float64)
state = np.zeros((H, W), dtype=np.uint8)  # 0=FAR, 1=TRIAL, 2=KNOWN

# heap：优先队列，装 (T, r, c)，每次弹出全局最小 T 的 TRIAL 点，设为 KNOWN。这是 FMM/Dijkstra 的加速核心：从 O(N²) 降到 O(N log N)。
heap = []  # (T_ij, r, c)

def neighbors4(r, c):
    for dr, dc in ((-1,0), (1,0), (0,-1), (0,1)):
        rr, cc = r+dr, c+dc
        if 0 <= rr < H and 0 <= cc < W:
            yield rr, cc

def update_T(r, c):
    """ Godunov upwind 局部解（标准FMM更新），单元格间距=1 """
    if F[r, c] <= 0.0:  # 障碍或不可通行
        return INF
    # 取相邻已知/当前值最小的 T
    Tx = []
    Ty = []
    # 上下
    for rr, cc in ((r-1, c), (r+1, c)):
        if 0 <= rr < H:
            Tx.append(T[rr, c])
    # 左右
    for rr, cc in ((r, c-1), (r, c+1)):
        if 0 <= cc < W:
            Ty.append(T[r, cc])
    ax = min(Tx) if Tx else INF
    ay = min(Ty) if Ty else INF

    a = min(ax, ay)
    b = max(ax, ay)
    invF = 1.0 / F[r, c]

    # 参考 Osher–Sethian FMM 更新
    if abs(a - b) >= invF:
        T_new = a + invF
    else:
        # 解 (T - a)^2 + (T - b)^2 = invF^2
        # 推导可得：
        T_new = 0.5 * (a + b + np.sqrt(max(0.0, 2.0*invF*invF - (a - b)*(a - b))))
    return min(T_new, T[r, c])

# 初始化：从 goal 处出发
T[goal] = 0.0
state[goal] = KNOWN
# 将 goal 的可行邻居压入 TRIAL
for rr, cc in neighbors4(*goal):
    if free[rr, cc] == 1 and state[rr, cc] != KNOWN:
        T[rr, cc] = update_T(rr, cc)
        state[rr, cc] = TRIAL
        heapq.heappush(heap, (T[rr, cc], rr, cc))

# 主循环
while heap:
    Tij, r, c = heapq.heappop(heap)
    if state[r, c] == KNOWN:
        continue
    state[r, c] = KNOWN
    if (r, c) == start:
        break  # 已经到达 start，T(start) 已确定为最短代价
    # 松弛邻居
    for rr, cc in neighbors4(r, c):
        if free[rr, cc] == 0:
            continue
        if state[rr, cc] != KNOWN:
            Told = T[rr, cc]
            Tnew = update_T(rr, cc)
            if Tnew < Told:
                T[rr, cc] = Tnew
                state[rr, cc] = TRIAL
                heapq.heappush(heap, (Tnew, rr, cc))

# ----------------------------
# 4) 最陡下降回溯（8邻域贪心下降 T）
# ----------------------------
def neighbors8(r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r+dr, c+dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr, cc

if not np.isfinite(T[start]):
    raise RuntimeError("起点不可达（或被障碍隔开）。")

path = [start]
cur = start
max_steps = H * W  # 极限保护
for _ in range(max_steps):
    if cur == goal:
        break
    # 在 8 邻域里找 T 最小的点（必须下降）
    r, c = cur
    best = None
    bestT = T[r, c]
    for rr, cc in neighbors8(r, c):
        if T[rr, cc] < bestT:
            bestT = T[rr, cc]
            best = (rr, cc)
    if best is None:
        # 已经无法下降，接近极小（通常会离 goal 很近）
        break
    path.append(best)
    cur = best

# ----------------------------
# 5) 结果与可视化
# ----------------------------
K_est = T[start]  # 这就是 K(x,y) 的数值近似
print(f"Estimated minimal cost K(x,y) = {K_est:.6f}")
print(f"Path length (in steps) = {len(path)}")

# 可视化：
#   背景：grid
#   等势线：T 的若干等高线（帮助理解“地形”）
#   路径：从 start 到 goal 的折线
plt.figure(figsize=(10, 8))
plt.imshow(grid, origin="upper")
# 画 T 的等高线（只在可行域画）
levels = np.linspace(0, np.nanpercentile(T[np.isfinite(T)], 90), 15)
T_show = np.where(np.isfinite(T), T, np.nan)
plt.contour(T_show, levels=levels, linewidths=0.7)
# 路径
py, px = zip(*path)  # row列为y，col列为x
plt.plot(px, py, linewidth=2.0)
# 起点/终点
plt.scatter([start[1], goal[1]], [start[0], goal[0]], s=30, marker="o")
plt.title("Fast Marching minimal path")
plt.tight_layout()
plt.show()
