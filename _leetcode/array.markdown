---
layout: post
title: 数组篇
---

总目录：

- 矩阵
  - 遍历
  - 动态规划
- 回溯
  - 全组合
  - 全排列
- 哈希表
- 多状态缓存



### 31. Next permutation

核心方法：`遍历`/`逆序对`/`顺序对`

解题思路：

1. permutation的方法就是交换数组中的一对元素，交换`逆序对`只会更小，因此必须交换`顺序对`
2. 越靠右的改变对数组值的影响越小，因此要查找"最靠右"的`顺序对`，即其起点和终点都是最靠右的
3. 最靠右的起点就是从右往左第一个下坡点，最靠右的终点是比它大的数里面的最小值，即从右到左第一个比起点大的数
4. 为了让结果进一步缩小，还要将上述起点右侧的逆序子数组全部反转

```python
"""
实现获取 下一个排列 的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须 原地 修改，只允许使用额外常数空间。

链接：https://leetcode-cn.com/problems/next-permutation
"""
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        # 1. 找到最靠右的顺序对
        lower = None
        higher = None
        # 1.1 先找左端点
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] < nums[i+1]:
                lower = i
                break
        # 1.2 再找右端点
        if lower is not None:
            for i in range(len(nums) - 1, lower, -1):
                if nums[i] > nums[lower]:
                    higher = i
                    break
        else:
            # 整个序列是降序排列的，不存在任何顺序对
            self.reverse(nums, 0, len(nums) - 1)
            return

        # 2. 交换最右顺序对
        nums[lower], nums[higher] = nums[higher], nums[lower]

        # 3. 升序排列lower右侧的数
        # 交换后，lower右侧一定全是逆序对，否则lower不可能是交换前最右侧的顺序对
        self.reverse(nums, lower + 1, len(nums) - 1)
    
    # 反转start到end之间的部分
    def reverse(self, nums, start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
```





### 532. K-diff pairs in an array

核心方法：

解题思路：

```python
"""
给定一个整数数组和一个整数 k，你需要在数组里找到 不同的 k-diff 数对，并返回不同的 k-diff 数对 的数目。

这里将 k-diff 数对定义为一个整数对 (nums[i], nums[j])，并满足下述全部条件：

0 <= i < j < nums.length
|nums[i] - nums[j]| == k
注意，|val| 表示 val 的绝对值。

链接：https://leetcode-cn.com/problems/k-diff-pairs-in-an-array
"""
class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        # 本题跟滑窗没什么关系，主要是利用hash
        num_count = {}
        for num in nums:
            num_count[num] = num_count.get(num, 0) + 1

        count = 0
        if k == 0:
            for num, cnt in num_count.items():
                if cnt > 1:
                    count += 1
        else:
            for num, _ in num_count.items():
                if num + k in num_count:
                    count += 1
        
        return count
```





### 48. Rotate image

核心方法：`矩阵遍历`

解题思路：

1. 矩阵有三种翻转：对角线翻转（转置），水平翻转，垂直翻转
2. 顺时针旋转 = 对角线翻转 + 垂直翻转

```python
"""
给定一个n × n的二维矩阵matrix表示一个图像。请你将图像顺时针旋转90度。

你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。

链接：https://leetcode-cn.com/problems/rotate-image
"""
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        row_count = len(matrix)
        col_count = len(matrix[0])

        # transpose
        for i in range(row_count):
            for j in range(i + 1, col_count):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # vertically reverse
        for i in range(row_count):
            for j in range(col_count // 2):
                matrix[i][j], matrix[i][col_count - 1 - j] = matrix[i][col_count - 1 - j], matrix[i][j]
```



### 56. Merge intervals

核心方法：

解题思路：按左边界排序后就很直观了

```python
"""
以数组intervals表示若干个区间的集合，其中单个区间为intervals[i] = [starti, endi]。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

链接：https://leetcode-cn.com/problems/merge-intervals
"""
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 按照左边界排序
        intervals.sort(key=lambda x: x[0])
        
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                if merged[-1][0] <= interval[0] <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], interval[1])
                else:
                    merged.append(interval)
        
        return merged
```





### 73. Set matrix zeroes

核心方法：

解题思路：

```python
"""
给定一个m x n的矩阵，如果一个元素为0，则将其所在行和列的所有元素都设为0。请使用原地算法。

进阶：
一个直观的解决方案是使用 O(mn) 的额外空间，但这并不是一个好的解决方案。
一个简单的改进方案是使用 O(m + n) 的额外空间，但这仍然不是最好的解决方案。
你能想出一个仅使用常量空间的解决方案吗？

链接：https://leetcode-cn.com/problems/set-matrix-zeroes
"""
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        if not matrix:
            return

        rows = []
        cols = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    rows.append(i)
                    cols.append(j)
        
        for row in rows:
            for i in range(len(matrix[0])):
                matrix[row][i] = 0
        for col in cols:
            for i in range(len(matrix)):
                matrix[i][col] = 0
```















### 128. Longest consecutive sequence

核心方法：`哈希表`

解题思路：

```python
"""
给定一个未排序的整数数组nums，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

进阶：你可以设计并实现时间复杂度为 O(n) 的解决方案吗？

链接：https://leetcode-cn.com/problems/longest-consecutive-sequence
"""
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        max_len = 0
        # key所在的连续子序列长度
        seq_lens = {}
        for num in nums:
            # 从未出现过的新数字
            if num not in seq_lens:
                # 查看左右相邻区间的长度（可能不存在）
                left_length = seq_lens.get(num - 1, 0)
                right_length = seq_lens.get(num + 1, 0)
                
                # 计算新数字所在区间长度
                cur_length = left_length + 1 + right_length
                max_len = max(max_len, cur_length)
                
                # 更新长度
                seq_lens[num] = cur_length
                # 只有边界点影响计算，因此可以只更新边界点记录的区间长度
                seq_lens[num - left_length] = cur_length
                seq_lens[num + right_length] = cur_length
        return max_len
```





### 229. Majority element II

核心方法：

解题思路：最多有2个数超过N/3

```python
"""
给定一个大小为n的整数数组，找出其中所有出现超过⌊ n/3 ⌋次的元素。

进阶：尝试设计时间复杂度为O(n)、空间复杂度为O(1)的算法解决此问题。

链接：https://leetcode-cn.com/problems/majority-element-ii
"""
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        candidates = {}

        # finding
        i = 0
        while len(candidates) < 2 and i < len(nums):
            candidates[nums[i]] = candidates.get(nums[i], 0) + 1
            i += 1
        while i < len(nums):
            if nums[i] in candidates:
                candidates[nums[i]] = candidates[nums[i]] + 1
            else:
                all_positive = True
                for key in candidates:
                    if candidates[key] == 0:
                        all_positive = False
                        break
                # decreasing each by one
                if all_positive:
                    for key in candidates:
                        candidates[key] = candidates[key] - 1
                # or replace the zero valued one
                else:
                    for key in list(candidates):
                        if candidates[key] == 0:
                            del candidates[key]
                            break
                    candidates[nums[i]] = 1
            i += 1
        
        # counting
        result = []
        for key in candidates:
            count = 0
            for num in nums:
                if key == num:
                    count += 1
            if count > len(nums) // 3:
                result.append(key)

        return result
```



### 238. Product of array except self

核心方法：`遍历`，`多状态缓存`

解题思路：双向累乘

```python
"""
给你一个长度为n的整数数组nums，其中n > 1，返回输出数组output，其中output[i]等于nums中除nums[i]之外其余各元素的乘积。

链接：https://leetcode-cn.com/problems/product-of-array-except-self
"""
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # 此题的trick在于善用结果数组，并将左右两个累乘数组分先后两批计算
        result = [0] * len(nums)
        # 从左到右的累乘，result[i]表示i左侧所有元素的乘积
        result[0] = 1
        for i in range(1, len(nums)):
            result[i] = result[i - 1] * nums[i - 1]
        # 从右到左的累乘，acc表示i所有右侧元素的乘积
        acc = 1
        for i in range(len(nums) - 2, -1, -1):
            acc = acc * nums[i + 1]
            result[i] = result[i] * acc

        return result
```





