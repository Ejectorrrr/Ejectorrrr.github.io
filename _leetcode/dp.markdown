---
layout: post
title: 动态规划篇
---



### 5. Longest palindromic substring

核心方法：`二维动态规划`，`双指针`

解题思路：

```python
"""
给你一个字符串 s，找到 s 中最长的回文子串。

链接：https://leetcode-cn.com/problems/longest-palindromic-substring/
"""
class Solution:
    """
    重点是在棋盘中遍历的方向
    """
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return None

        dp = [[0 for j in range(len(s))] for i in range(len(s))]
        max_len = 1
        max_start, max_end = 0, 0
        # end从左到右
        for end in range(len(s)):
            # start从end到左
            for start in range(end, -1, -1):
                if s[start] == s[end]:
                    dp[start][end] = dp[start + 1][end - 1] if end - start > 1 else 1
                else:
                    dp[start][end] = 0

                if dp[start][end] == 1:
                    if end - start + 1 > max_len:
                        max_start = start
                        max_end = end
                        max_len = end - start + 1

        return s[max_start:(max_end + 1)]
```



### 91. Decode ways

核心方法：

解题思路：dp[i]表示以i结尾的字符串有多少种表示方法

```python
"""
一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：

'A' -> 1
'B' -> 2
...
'Z' -> 26
要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：

"AAJF" ，将消息分组为 (1 1 10 6)
"KJF" ，将消息分组为 (11 10 6)
注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。

给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。

题目数据保证答案肯定是一个 32 位 的整数。

链接：https://leetcode-cn.com/problems/decode-ways
"""
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0':
            return 0

        # 初始化状态数组
        dp = [0 for _ in range(len(s))]
        dp[0] = 1
        for i in range(1, len(s)):
            if int(s[i]) > 0:
                dp[i] += dp[i - 1]
            if int(s[i - 1]) == 1 or (int(s[i - 1]) == 2 and int(s[i]) <= 6):
                dp[i] += dp[i - 2] if i > 1 else 1

        return dp[-1]
```



### 300. Longest increasing subsequence

核心方法：`一维动态规划`，`双指针`

解题思路：__dp[i]__ 表示以 __i__ 结尾的子序列中最多的顺序对数

```python
"""
给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

链接：https://leetcode-cn.com/problems/longest-increasing-subsequence
"""
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```



### 62. Unique paths

核心方法：`暴力`，`棋盘动态规划`，`memoization`

解题思路：棋盘动态规划是指一类特殊的动态规划，用二维矩阵保存状态，从右小角向左上角更新状态，每次只能朝左或朝上移动

```python
"""
一个机器人位于一个m x n网格的左上角（起始点在下图中标记为“Start”）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

问总共有多少条不同的路径？

链接：https://leetcode-cn.com/problems/unique-paths
"""
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        """
        每一个点到达终点的路径数，等于其右侧点到达终点的路径数+下方点到达终点的路径数
        """
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



### 63. Unique paths II

核心方法：`暴力`，`棋盘动态规划`，`memoization`

解题思路：

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



### 64. Minimum path sum

核心方法：`暴力`，`棋盘动态规划`，`memoization`

解题思路：

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

### 152. Maximum product subarray

核心方法：`动态规划`

解题思路：在一次遍历中，状态及状态转移方程可以有多个，最终结果是多种状态的综合。

```python
"""
给你一个整数数组nums，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

链接：https://leetcode-cn.com/problems/maximum-product-subarray/
"""
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        # 按照最大子序列和的做法：
        #     设dp[i]是以第i个数字结尾的最大子数组乘积
        # 但当前位置的最优解未必是由前一个位置的最优解转移得到的！此题的变数在于：
        #     对于nums[i] > 0，希望dp[i - 1]是尽可能大的数
        #     对于nums[i] < 0，希望dp[i - 1]是尽可能小的数
        if len(nums) == 1:
            return nums[0]

        cur_min = [0 for _ in range(len(nums))]
        cur_max = [0 for _ in range(len(nums))]
        cur_min[0] = cur_max[0] = nums[0]
        max_prod = nums[0]
        for i in range(1, len(nums)):
            # 第i个位置能得到的最大值和最小值
            # 取决于nums[i]是整数还是负数，有不同的选择
            cur_min[i] = min(cur_min[i - 1] * nums[i], cur_max[i - 1] * nums[i], nums[i])
            cur_max[i] = max(cur_min[i - 1] * nums[i], cur_max[i - 1] * nums[i], nums[i])
            max_prod = max(max_prod, cur_min[i], cur_max[i])

        return max_prod
```

