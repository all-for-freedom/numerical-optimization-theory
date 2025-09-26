# -*- coding: utf-8 -*-
"""
File:        find_way.py
Description: 

Created on:  Fri Sep 26 2025 15:07:16
Author:      Shen Yang
University:  Hunan Normal University
Email:       yangshen@hunnu.edu.cn
"""

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import skfmm  # pip install scikit-fmm
import matplotlib.pyplot as plt

# === 读图并做 free/F 和 d0 一样 ===
img = Image.open("./case1/map1.png").convert("L")
# img = Image.open("./case2/map2.png").convert("L")
# img = Image.open("./case3/map3.png").convert("L")
arr = np.array(img)
grid = (arr > 128).astype(np.uint8)       # 1=free, 0=obstacle
H, W = grid.shape
free = (grid == 1)
padded = np.pad(free.astype(np.uint8), 1, mode="constant", constant_values=0)
dpad = distance_transform_edt(padded)
d0 = dpad[1:-1, 1:-1].astype(np.float64)

eps = 1.0
F = np.zeros_like(d0, dtype=np.float64)
F[free] = d0[free] + eps                  # 速度场，障碍处保持 0

# === 用 skfmm 直接解 T：travel_time ===
# 关键：phi 的 0 等值线是“源集合”。我们把终点像素设成负，其余为正，
#      这样 phi=0 的小轮廓就落在终点像素边界附近。

# (col, row)）  map 1
col_row_start = (1750, 200)   # (x=1750, y=200)
col_row_goal  = (3100, 1400)  # (x=3100, y=1400)

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
# col_row_start = (735, 601)   
# col_row_goal  = (85, 850)

start = (col_row_start[1], col_row_start[0])
goal  = (col_row_goal[1],  col_row_goal[0])

phi = np.ones((H, W), dtype=float)
phi[goal] = -1.0                           # 终点做为源（front 从这里向外推进）

# skfmm 用的是 F|∇T|=1，所以传入的 speed 就是 F
T = skfmm.travel_time(phi, speed=F, dx=1.0)  # 得到整幅图的 T 场
# 不可达或障碍的地方，T 会是 nan；这里换成 +inf
T = np.where(np.isfinite(T), T, np.inf)

# === 回溯（和你原来一样，用 8 邻域最陡下降）===
def neighbors8(r, c, H, W):
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==0 and dc==0: continue
            rr, cc = r+dr, c+dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr, cc

if not np.isfinite(T[start]):
    raise RuntimeError("起点不可达（或与终点不连通）。")

path = [start]
cur = start
for _ in range(H*W):
    if cur == goal: break
    r, c = cur
    best = None; bestT = T[r,c]
    for rr, cc in neighbors8(r,c,H,W):
        if T[rr,cc] < bestT:
            bestT = T[rr,cc]; best = (rr,cc)
    if best is None: break
    path.append(best); cur = best

print("K_est =", T[start], "steps =", len(path))

# === 可视化 ===
plt.figure(figsize=(10,8))
plt.imshow(grid, origin="upper", cmap="gray", vmin=0, vmax=1)
T_show = np.where(np.isfinite(T), T, np.nan)
levels = np.linspace(np.nanmin(T_show), np.nanpercentile(T_show, 90), 15)
plt.contour(T_show, levels=levels, linewidths=0.7)
py, px = zip(*path)
plt.plot(px, py, lw=2)
plt.scatter([start[1], goal[1]], [start[0], goal[0]], s=30, marker="o")
plt.title("minimal path")
plt.tight_layout()
plt.show()
