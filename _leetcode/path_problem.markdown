---
layout: post
title: 路径问题
---





参考链接：https://mp.weixin.qq.com/s?__biz=MzU4NDE3MTEyMA==&mid=2247485580&idx=1&sn=84c99a0a8ab7b543c3678db577309b97&chksm=fd9ca393caeb2a859fafb0cb12683669ed1a0086cb22e1eaaa9ec323e033ab2cf3a77dfc5561&scene=178&cur_album_id=1773144264147812354#rd



##### 问题分类：

- 求枚举数
- 求最小值



### 一、求枚举数问题：

#### 62. Unique paths

有两种遍历方向，取决于dp的定义是从[0, 0]出发到达[i, j]的路径数，还是从[i, j]出发到达[m - 1, n  - 1]的路径数

```python
"""
一个机器人位于一个m x n网格的左上角（起始点在下图中标记为“Start”）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

问总共有多少条不同的路径？

链接：https://leetcode-cn.com/problems/unique-paths
"""
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        path_count = [[0 for _ in range(n)] for _ in range(m)]
        # 终点到达自己有1条路径
        path_count[-1][-1] = 1

        # 从终点开始向起点回溯（为了避免后效性，和查找回文串时同理）
        for row in range(m - 1, -1, -1):
            for col in range(n - 1, -1, -1):
                # 最下边一行
                if row == m - 1 and col < n - 1:
                    path_count[row][col] = path_count[row][col + 1]
                # 最右边一列
                if row < m - 1 and col == n - 1:
                    path_count[row][col] = path_count[row + 1][col]
                if row < m - 1 and col < n - 1:
                    path_count[row][col] = path_count[row + 1][col] + path_count[row][col + 1]

        return path_count[0][0]
```

#### 63. Unique paths II

```python
"""
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

链接：https://leetcode-cn.com/problems/unique-paths-ii
"""
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if obstacleGrid[-1][-1] == 1:
            return 0

        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    obstacleGrid[i][j] = -1
        obstacleGrid[-1][-1] = 1

        for row in range(m - 1, -1, -1):
            for col in range(n - 1, -1, -1):
                if obstacleGrid[row][col] == -1:
                    obstacleGrid[row][col] = 0
                    continue
                # 最下边一行
                if row == m - 1 and col < n - 1:
                    obstacleGrid[row][col] = obstacleGrid[row][col + 1]
                # 最右边一列
                if row < m - 1 and col == n - 1:
                    obstacleGrid[row][col] = obstacleGrid[row + 1][col]
                if row < m - 1 and col < n - 1:
                    obstacleGrid[row][col] = obstacleGrid[row + 1][col] + obstacleGrid[row][col + 1]

        return obstacleGrid[0][0]
```



### 二、求最小值问题

#### 64. Minimum path sum

```python
"""
给定一个包含非负整数的m x n网格grid，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

链接：https://leetcode-cn.com/problems/minimum-path-sum
"""
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        # 从右下角往左上角遍历，避免后效性
        for row in range(m - 1, -1, -1):
            for col in range(n - 1, -1, -1):
                # 最后一排
                if row == m - 1 and col < n - 1:
                    grid[row][col] = grid[row][col] + grid[row][col + 1]
                # 最后一列
                if row < m - 1 and col == n - 1:
                    grid[row][col] = grid[row][col] + grid[row + 1][col]
                if row < m - 1 and col < n - 1:
                    grid[row][col] = grid[row][col] + min(grid[row + 1][col], grid[row][col + 1])

        return grid[0][0]
```

#### 120. Triangle

```python
"""
给定一个三角形 triangle ，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

链接：https://leetcode-cn.com/problems/triangle
"""
class Solution:
    # 空间复杂度O(N^2)
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        for row in range(1, len(triangle)):
            for col in range(len(triangle[row])):
                triangle[row][col] += min(
                    triangle[row - 1][col] if col < cols - 1 else float('inf'),
                    triangle[row - 1][col - 1] if col > 0 else float('inf'))
        
        return min(triangle[-1])

    # 空间复杂度O(N)
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # 与最后一行的列数一致
        dp = [float('inf')] * len(triangle[-1])
        for j in range(len(triangle[0])):
            dp[j] = triangle[0][j]

        for i in range(1, len(triangle)):
            for j in range(len(triangle[i]) - 1, -1, -1):
                dp[j] = triangle[i][j] + min(dp[j], dp[j - 1] if j > 0 else float('inf'))
        
        return min(dp)
```

#### 931. Minimum falling path sum

```python
"""
给你一个 n x n 的 方形 整数数组 matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和 。

下降路径 可以从第一行中的任何元素开始，并从每一行中选择一个元素。在下一行选择的元素和当前行所选元素最多相隔一列（即位于正下方或者沿对角线向左或者向右的第一个元素）。具体来说，位置 (row, col) 的下一个元素应当是 (row + 1, col - 1)、(row + 1, col) 或者 (row + 1, col + 1) 。

链接：https://leetcode-cn.com/problems/minimum-falling-path-sum
"""
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        # 从第一行任意点开始到达[i, j]的最小路径和
        dp = [[float('inf') for _ in range(n)] for _ in range(m)]
        for j in range(n):
            dp[0][j] = matrix[0][j]

        for i in range(1, m):
            for j in range(n):
                dp[i][j] = (matrix[i][j] + 
                            min(dp[i - 1][j - 1] if j > 0 else float('inf'), 
                                dp[i - 1][j], 
                                dp[i - 1][j + 1] if j < n - 1 else float('inf')))
        return min(dp[-1])
```

