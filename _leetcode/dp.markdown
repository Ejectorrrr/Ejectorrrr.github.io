---
layout: post
title: 动态规划篇
---







基础数据结构：线性 -> 树 -> 图

线性数据结构的遍历：顺序（单指针、双指针）、二分

非线性数据结构的遍历：DFS、BFS、分治

二分若特指能够计算出中点索引的情况，分治就是对二分在非线性数据结构上的拓展

动态规划、回溯、贪心只是构筑在数据结构及其遍历方法之上的优化解法



##### 基本概念：

- 无后效性
- memoization



##### 专题：

- [背包问题]({% link _leetcode/package_problem.markdown %})
- [路径问题]({% link _leetcode/path_problem.markdown%})
- 字符串匹配



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



### 518. Coin change 2

核心方法：

解题思路：

```python
"""
给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。

请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。

假设每一种面额的硬币有无限个。 

题目数据保证结果符合 32 位带符号整数。

链接：https://leetcode-cn.com/problems/coin-change-2
"""
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        # 金额i对应的组合数
        dp = [0] * (amount + 1)
        dp[0] = 1

        # 内外循环如果互换，则会出现重复组合
        for coin in coins:
            # 按顺序出每枚硬币时，所能形成的面值的组合数
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]

        return dp[-1]
```



### 70. Climbing stairs

核心方法：

解题思路：与Fibonacci数组一样，分治法存在大量重复计算，使用memoization优化的空间复杂度为O(N)，最优解法是将自顶向下的递归转化为自底向上的迭代

```python
"""
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
注意：给定 n 是一个正整数。

链接：https://leetcode-cn.com/problems/climbing-stairs/
"""
class Solution:
    def climbStairs(self, n: int) -> int:
        # 初始状态，分别对应n=0和n=1时的结果
        a = 1
        b = 1
        for _ in range(n):
            a, b = b, a + b

        return a
```



### 72. Edit distance

核心方法：

解题思路：

```python
"""
给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：
- 插入一个字符
- 删除一个字符
- 替换一个字符

链接：https://leetcode-cn.com/problems/edit-distance
"""
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # d[i][j]表示word1的前i位和word2的前j位之间的编辑距离
        d = [[0 for _ in range(len(word2) + 1)] for _ in range(len(word1) + 1)]

        # 初始化
        for i in range(len(word1) + 1):
            d[i][0] = i
        for j in range(len(word2) + 1):
            d[0][j] = j

        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i - 1] == word2[j - 1]:
                    d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1])
                else:
                    d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
        
        return d[-1][-1]
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



### 1143. Longest common subsequence

核心方法：

解题思路：

```python
"""
给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

链接：https://leetcode-cn.com/problems/longest-common-subsequence
"""
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # text1的前i子串与text2的前j子串间的最长公共子序列
        # 边界都是0，因此不需要专门处理
        dp = [[0 for _ in range(len(text2) + 1)] for _ in range(len(text1) + 1)]

        for i in range(1, len(text1) + 1):
            for j in range(1, len(text2) + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[-1][-1]
```

