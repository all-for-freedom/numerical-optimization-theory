Given two points on the two-dimensional plane, find a path to avoid all obstacles, so that the integral of the minimum distance from each point to the obstacle on the path is the smallest.

# 数学支撑（代码对应的理论与结果）

## **1. 路径代价的数学模型**

目标是求解最小代价函数：
$$
K(x,y) \;=\; \inf_{\gamma:x\to y} \int_{\gamma} \frac{1}{d_0(z)} \, ds
$$
其中

- $\gamma$ 是从 x 到 y 的任意可行路径；
- $d_0(z)$ 是点 z 到最近障碍的欧氏距离；
- 积分里 $\tfrac{1}{d_0(z)}$ 是“局部权重”，距离障碍越近，代价越大。

这就是代码里 $K_{est} = T[start]$ 的数学定义。

## **2. 欧氏距离变换 (EDT)**

对二值图（1=free, 0=obstacle），欧氏距离场定义为：
$$
d_0(p) \;=\; \min_{q \in \mathcal{O}} \|p - q\|_2,
$$
其中 $\mathcal{O}$ 是障碍集合。

## **3. 速度场定义**

速度场设为：
$$
F(p) \;=\; d_0(p) + \varepsilon, \quad F(p)=0 \;\;\text{if } p \in \mathcal{O}.
$$
这里 $\varepsilon>0$ 是数值稳定项，避免除零。
所以：
- 离障碍越近，F 越小，传播速度越慢。
- 在障碍上，F=0，不可通行。
## **4. Eikonal 方程**

最优代价函数 T 满足 **Eikonal 方程**：
$$
|\nabla T(p)| = \frac{1}{F(p)}, \qquad T(y)=0.
$$
这就是代码里 T[r,c] 数组的数学定义。
物理意义：从任意点 p 到目标点 y 的最小旅行时间。
## **5. Godunov 上风差分格式**
离散化 PDE，得到 **局部更新公式**：
设某格点 (i,j) 的水平邻居最小值 a，垂直邻居最小值 b，则 T_{ij} 满足：
$$
\begin{cases} T_{ij} = \min(a,b) + \frac{1}{F_{ij}}, & |a-b| \ge \frac{1}{F_{ij}}, \\ T_{ij} = \frac{a+b}{2} + \frac{1}{2}\sqrt{\frac{2}{F_{ij}^2} - (a-b)^2}, & |a-b| < \frac{1}{F_{ij}}. \end{cases}
$$
## **6. Fast Marching Method (FMM)**

FMM 的因果推进原则：

- 从目标点 y 开始，初始化 T(y)=0。
- 使用 **堆 (min-heap)** 维护所有待更新点的候选值。
- 每次弹出最小 T 的点，设为 **KNOWN**（确定值）。
- 用该点更新邻居，放入堆。
- 直到起点 x 被确定。

复杂度：
$$
O(N \log N) \quad \text{（N = 可行域像素数）}.
$$
## **7. 最优路径回溯**

路径是 **最陡下降线**：
$$
\dot{\gamma}(t) = -\frac{\nabla T(\gamma(t))}{|\nabla T(\gamma(t))|},
$$
即从起点出发，沿梯度负方向下降，直到到达目标点。
代码里用 **8 邻域贪心下降**近似实现。