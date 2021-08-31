---
layout: post
title: 背包问题
---





参考链接：https://mp.weixin.qq.com/s/xmgK7SrTnFIM3Owpk-emmg



背包问题是 __动态规划__ 中的一类典型问题，对背包问题的抽象：

- 给定素材的 __成本__ 和 __价值__，如何在 __成本边界__ 内实现 __价值最大化__

背包问题本质上属于组合优化的 __NP完全问题__，只能通过 __穷举__ + __验证__ 的方式求解。

既然是穷举，最直接的手段是DFS，而dp的转移方程也是由DFS过程推导出来的。



## 01背包问题

##### 经典题设：

```python
"""
有N件物品，每件物品的体积和价值分别对应数组v和w，有一个容积为V的背包，求该背包能容纳物品价值之和的最大值
"""
def solution(v: List[int], w: List[int], V: int) -> int:
    pass
```

所谓 __“01”__ 是指对每件物品只有选或不选两种可能。



__优化过程__：DFS -> 二维动态规划 -> 滚动数组 -> 一维动态规划

1）DFS：

可以采用前序或后序遍历，下面采用了后序遍历

```python
def dfs(v: List[int], w: List[int], i: int, c: int) -> int:
    """
    i: 当前待遍历物品
    c: 当前剩余容量

    return: 遍历完当前及之后所有物品所能获得的最大价值
    """
    if i >= len(v) or c <= 0:
        return 0
    
    # 纳入当前物品
    value1 = dfs(v, w, i + 1, c - v[i]) + w[i]
    # 不纳入当前物品
    value2 = dfs(v, w, i + 1, c)
    
    return max(value1, value2)
```

2）二维动态规划

通过DFS可以发现，__成本__ 和 __成本边界__ 是主动变化的，__价值__ 是被动变化的，因此将物品数量和背包容量作为状态矩阵的行和列，最大价值作为状态值

```python
def dp2D(v: List[int], w: List[int], V: int) -> int:
    n = len(v)
    # 考虑前i个物品、总容量为c的情况下所能获得的最大价值
    dp = [[0 for _ in range(V + 1)] for _ in range(n + 1)]
    # 首行首列的初始值都是0
    for i in range(1, n + 1):
        for j in range(1, V + 1):
            # 纳入当前物品
            value1 = dp[i - 1][j - v[i - 1]] + w[i - 1] if j >= v[i - 1] else 0
            # 不纳入当前物品
            value2 = dp[i - 1][j]
            dp[i][j] = max(value1, value2)
    return dp[-1][-1]
```

3）滚动数组

二维动态规划是逐行更新的，且当前行只与上一行有关，因此可以将状态矩阵进一步压缩为只有两行

```python
def dp2D(v: List[int], w: List[int], V: int) -> int:
    n = len(v)
    dp = [[0 for _ in range(V + 1)] for _ in range(2)]
    for i in range(1, n + 1):
        for j in range(1, V + 1):
            value1 = dp[(i - 1) % 2][j - v[i - 1]] + w[i - 1] if j >= v[i - 1] else 0
            value2 = dp[(i - 1) % 2][j]
            dp[i % 2][j] = max(value1, value2)
    return dp[(n - 1) % 2][-1]
```

4）一维动态规划

若原地更新状态值，则可将状态矩阵进一步压缩为一维。考虑到更新过程中，后面的值会不断依赖前面的值，因此不能从0到V遍历，必须从V到0遍历

```python
def dp1D(v: List[int], w: List[int], V: int) -> int:
    n = len(v)
    # 初始状态不纳入任何物品，因此最大价值都是0
    dp = [0] * (V + 1)
    for i in range(1, n + 1):
        for j in range(V， 0， -1):
            value1 = dp[j - v[i - 1]] if j >= v[i - 1] else 0
            value2 = dp[j]
            dp[j] = max(value1, value2)
    return dp[-1]
```

总结：

- 上述各方案的计算复杂度始终是 $O(N^2)$​​，空间复杂度从 $O(N^2)$ 优化为 $O(N)$​​​
- 素材数量决定了行数，成本边界决定了列数
- dp的状态数组中存储的一般是某种 __统计聚合结果__（对若干DFS路径），例如最值、加和等



#### 416. Partition equal subset sum

解题思路：将问题转化为 `01背包问题`，

- __成本__ 和 __价值__ 都是 数组中各数字的值（最大成本和最大价值一致），__成本边界__ 是 数组和/2
- 总行数是 数组长度（素材数量），总列数是 数组和/2（成本边界）

```python
"""
给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

链接：https://leetcode-cn.com/problems/partition-equal-subset-sum/
"""
class Solution:
    """
    解法一：间接求解
    """
    def canPartition(self, nums: List[int]) -> bool:
        # 不能少于2个元素
        if len(nums) < 2:
            return False
		# 数组和不能为奇数
        if sum(nums) % 2 == 1:
            return False

        n = len(nums)
        boundary = sum(nums) // 2
        # 前i个数字总成本不超过j的最大价值
        dp = [[0] * (boundary + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, boundary + 1):
                value1 = dp[i-1][j-nums[i-1]] + nums[i-1] if j >= nums[i-1] else 0
                value2 = dp[i-1][j]
                dp[i][j] = max(value1, value2)

    	# 在成本边界处若最大价值也是 数组和/2，则存在解（本题最大价值不会比成本边界更大）
        return dp[-1][-1] == boundary

    """
    解法二：直接求解
    """
	def canPartition(self, nums: List[int]) -> bool:
        if len(nums) < 2:
            return False
        if sum(nums) % 2 == 1:
            return False

        n = len(nums)
        boundary = sum(nums) // 2
        # 前i个数字是否可以组成总和j
        dp = [[False] * (boundary + 1) for _ in range(n + 1)]

        # 初始化
        dp[0][0] = True

        for i in range(1, n + 1):
            for j in range(1, boundary + 1):
                value1 = dp[i - 1][j - nums[i - 1]] if j >= nums[i - 1] else False
                value2 = dp[i - 1][j]
                dp[i][j] = value1 or value2

        return dp[-1][-1]
```



## 完全背包问题

##### 经典题设：

```python
"""
有N种物品，每种物品都有无限件，每种物品的体积和价值分别对应数组v和w，有一个容积为C的背包，求该背包能容纳物品价值之和的最大值
"""
def solution(v: List[int], w: List[int], C: int) -> int:
    pass
```

相比01背包问题：每件物品可以选择多次



1）二维动态规划

```python
def dp2D(v: List[int], w: List[int], C: int) -> int:
    n = len(v)
    # 考虑前i种物品、总容量为c的情况下所能获得的最大价值
    dp = [[0] * (C + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, C + 1):
            k = 0
            while True:
                if j < k * v[i - 1]:
                    break
                dp[i][j] = max(dp[i][j], dp[i-1][j-k*v[i-1]] + k*w[i-1])
                k += 1
    
    return dp[-1][-1]
```

2）滚动数组

```python
def dp2D(v: List[int], w: List[int], C: int) -> int:
    n = len(v)
    dp = [[0] * (C + 1) for _ in range(2)]
    
    for i in range(1, n + 1):
        for j in range(1, C + 1):
            k = 0
            while True:
                if j < k * v[i - 1]:
                    break
                dp[i%2][j] = max(dp[i%2][j], dp[(i-1)%2][j-k*v[-1]] + k*w[i-1])
                k += 1

    return dp[n%2][-1]
```

3）一维动态规划

与01背包问题的主要区别是，j从小到大遍历

```python
def dp1D(v: List[int], w: List[int], C: int) -> int:
    n = len(v)
    dp = [0] * (C + 1)
    
    for i in range(1, n + 1):
        for j in range(1, C + 1):
            dp[j] = max(dp[j], dp[j - v[i - 1]] + w[i - 1] if j >= v[i - 1] else 0)
	
    return dp[-1]
```



#### 279. 

