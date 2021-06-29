---
layout: post
title: 数组篇
---

总目录：

- 双指针
  - 夹逼
  - partition
- 二分
  - 经典
  - 二叉树
- divide and conquer
- 矩阵
  - 遍历
  - 动态规划
- 回溯
  - 全组合
  - 全排列
- 哈希表
- 多状态缓存



### 15. 3Sum

核心方法：`双指针夹逼`

解题思路：固定任意一个元素（$O(N)$），其余两个元素按照双指针夹逼方式移动（$O(N)$），总体是$O(N^2)$时间复杂度。

```python
"""
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。

链接：https://leetcode-cn.com/problems/3sum
"""
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 排序：1. 便于跳过重复；2. 便于使用双指针
        sorted_nums = sorted(nums)

        results = []
        # 固定任意一个元素，其余两个元素按照双指针模式联动
        for first in range(len(sorted_nums) - 2):
            # 跳过重复
            if first > 0 and sorted_nums[first] == sorted_nums[first - 1]:
                continue
            
            # 2Sum问题
            second = first + 1
            third = len(sorted_nums) - 1
            target = -sorted_nums[first]
            while second < third:
                # 跳过重复
                if second > first + 1 and sorted_nums[second] == sorted_nums[second - 1]:
                    second += 1
                    continue
                if third < len(sorted_nums) - 1 and sorted_nums[third] == sorted_nums[third + 1]:
                    third -= 1
                    continue

                if sorted_nums[second] + sorted_nums[third] == target:
                    results.append([sorted_nums[first], sorted_nums[second], sorted_nums[third]])
                    second += 1
                    third -= 1
                elif sorted_nums[second] + sorted_nums[third] > target:
                    third -= 1
                else:
                    second += 1
        
        return results
```



### 16. 3Sum closest

核心方法：`双指针夹逼`

解题思路：固定第一个元素，其余两个元素按照双指针夹逼方式移动，移动过程中不断更新最靠近target的和。

```python
"""
给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

链接：https://leetcode-cn.com/problems/3sum-closest
"""
import math

class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # 排序：1. 便于跳过重复；2. 便于使用双指针
        nums = sorted(nums)
        distance = math.inf
        result = None
        for first in range(len(nums) - 2):
            # 跳过重复
            if first > 0 and nums[first] == nums[first - 1]:
                continue
            
            # 2Sum问题
            second = first + 1
            third = len(nums) - 1
            while second < third:
                # 跳过重复
                if second > first + 1 and nums[second] == nums[second - 1]:
                    second += 1
                    continue
                if third < len(nums) - 1 and nums[third] == nums[third + 1]:
                    third -= 1
                    continue
                
                sum_ = nums[first] + nums[second] + nums[third]
                if abs(target - sum_) < distance:
                    distance = abs(target - sum_)
                    result = sum_
                
                if sum_ == target:
                    return result
                elif sum_ < target :
                    second += 1
                else:
                    third -= 1
        
        return result
```



### 18. 4Sum

核心方法：`双指针夹逼`

解题思路：固定前两个元素（$O(N^2)$），剩下两个元素按照双指针夹逼方式移动（$O(N)$），因此总体是$O(N^3)$复杂度。

```python
"""
给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
注意：答案中不可以包含重复的四元组。

链接：https://leetcode-cn.com/problems/4sum
"""
class Solution:
    # 复杂度O(N3)：两层外循环后使用双指针
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        # 1. 排序
        nums = sorted(nums)
        # 2. 双指针
        results = []
        for first in range(len(nums) - 3):
            # 跳过重复
            if first > 0 and nums[first] == nums[first - 1]:
                continue
            for second in range(first + 1, len(nums)- 2):
                # 跳过重复
                if second > first + 1 and nums[second] == nums[second - 1]:
                    continue
                # 2Sum
                third = second + 1
                fourth = len(nums) - 1
                while third < fourth:
                    if third > second + 1 and nums[third] == nums[third - 1]:
                        third += 1
                        continue
                    if fourth < len(nums) - 1 and nums[fourth] == nums[fourth + 1]:
                        fourth -= 1
                        continue

                    sum_ = nums[first] + nums[second] + nums[third] + nums[fourth]
                    if sum_ == target:
                        results.append([nums[first], nums[second], nums[third], nums[fourth]])
                        third += 1
                        fourth -= 1
                    elif sum_ < target:
                        third += 1
                    else:
                        fourth -= 1
        
        return results
```



#### 2Sum模板：

```python
start = 0
end = len(nums) - 1
results = []
while start < end:
    sum_ = nums[start] + nums[end]
    if sum_ == target:
        results.append([start, end])
        start += 1
        end -= 1
    elif sum_ < target:
        start += 1
    else:
        end -= 1
return results
```

-----



### 11. Container with most water

核心方法：`双指针夹逼`

解题思路：从最大宽度开始，在缩短宽度的同时应尽量提升高度，因此每次移动短板并更新新的容积。

```python
"""
给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

链接：https://leetcode-cn.com/problems/container-with-most-water
"""
class Solution:
    def maxArea(self, height: List[int]) -> int:
        volume = 0
        # 从最大宽度开始
        left, right = 0, len(height) - 1
        while left < right:
            volume = max(volume, (right - left) * min(height[left], height[right]))
            # 只要减小宽度，必须试图用增加高度来补偿
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return volume
```



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



### 4. Median of two sorted arrays

核心方法：`二分`

解题思路：

```python
```



### 33. Search in rotated sorted array

核心方法：`二分`

解题思路：

1. 每轮都将数组分为 __有序部分__ 和 __无序部分__
2. 若target不在有序部分的值域内，则进入无序部分继续查找

```python
"""
整数数组nums按升序排列，数组中的值互不相同。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标从0开始计数）。例如， [0,1,2,4,5,6,7]在下标3处经旋转后可能变为[4,5,6,7,0,1,2]。

给你旋转后的数组nums和一个整数target，如果nums中存在这个目标值target，则返回它的下标，否则返回-1。

链接：https://leetcode-cn.com/problems/search-in-rotated-sorted-array
"""
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 任意一个mid，把nums分为两部分，有序部分和无序部分
        # 有序和无序部分，边界移动的逻辑不同
        start, end = 0, len(nums) - 1
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return mid

            # 判断哪一边是有序的:
            # 右侧有序
            if nums[mid] < nums[start]:
                # 在有序部分的值域内
                if nums[mid] < target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1
            # 左侧有序
            elif nums[mid] > nums[end]:
                # 在有序部分的值域内
                if nums[start] <= target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            # 都有序
            else:
                # 常规二分
                if nums[mid] < target:
                    start = mid + 1
                else:
                    end = mid - 1

        return -1
```



### 81. Search in rotated sorted array II

核心方法：`二分`

解题思路：

```python
"""
已知存在一个按非降序排列的整数数组nums，数组中的值不必互不相同。

在传递给函数之前，nums在预先未知的某个下标k（0 <= k < nums.length）上进行了旋转 ，使数组变为[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如，[0,1,2,4,4,4,5,6,6,7]在下标5处经旋转后可能变为[4,5,6,6,7,0,1,2,4,4] 。

给你旋转后的数组nums和一个整数target，请你编写一个函数来判断给定的目标值是否存在于数组中。如果nums中存在这个目标值target，则返回true，否则返回false。

链接：https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii
"""
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        start, end = 0, len(nums) - 1
        while start <= end:
            mid = start + (end - start) // 2
            # 特殊情况是nums[mid] == nums[start] == nums[end]
            if nums[mid] == target:
                return True

            if nums[mid] == nums[start] and nums[mid] == nums[end]:
                start += 1
                end -= 1
            # 左半边有序（注意是<=）
            elif nums[start] <= nums[mid]:
                if nums[start] <= target <= nums[mid]:
                    end = mid - 1
                else:
                    start += 1
            # 右半边有序（注意是<=）
            elif nums[mid] <= nums[end]:
                if nums[mid] <= target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1
        
        return False
```



### 153. Find minimum in rotated sorted array

核心方法：`二分`

解题思路：

```python
"""
已知一个长度为n的数组，预先按照升序排列，经由1到n次旋转后，得到输入数组。例如，原数组nums=[0,1,2,4,5,6,7]在变化后可能得到：
若旋转4次，则可以得到[4,5,6,7,0,1,2]
若旋转7次，则可以得到[0,1,2,4,5,6,7]
注意，数组[a[0], a[1], a[2], ..., a[n-1]]旋转一次的结果为数组[a[n-1], a[0], a[1], a[2], ..., a[n-2]]。

给你一个元素值互不相同的数组nums，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的最小元素。

链接：https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array
"""
class Solution:
    def findMin(self, nums: List[int]) -> int:
        start, end = 0, len(nums) - 1
        while start <= end:
            mid = start + (end - start) // 2
            # 命中条件
            if nums[start] <= nums[end]:
                return nums[start]

            # 因无序区间同时包含了数组最大值和最小值，未命中则始终寻找下一个无序区间
            # 左侧有序
            elif nums[start] <= nums[mid]:
                start = mid + 1
            # 右侧有序
            elif nums[mid] <= nums[end]:
                end = mid

        return -1
```



### 162. Find peak element

核心方法：`二分`

解题思路：不断寻找无序区间

```python
"""
峰值元素是指其值大于左右相邻值的元素。

给你一个输入数组nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。

你可以假设nums[-1] = nums[n] = -∞。

链接：https://leetcode-cn.com/problems/find-peak-element
"""
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        start, end = 0, len(nums) - 1
        while start < end:
            mid = start + (end - start) // 2
            # 因为mid总是更靠左，因此必须和右边比较，否则会越界
            # 如果mid靠右取，则应和左边比较
            if nums[mid] > nums[mid + 1]:
                end = mid
            else:
                start = mid + 1
        return start
```



### 34. Find first and last position of element in sorted array

核心方法：`二分`

解题思路：

1. 开头和结尾需要两次二分查找
2. 每次找到target后，需要判断是否还有重复元素

```python
"""
给定一个按照升序排列的整数数组nums，和一个目标值target。找出给定目标值在数组中的开始位置和结束位置。
如果数组中不存在目标值 target，返回[-1, -1]。

进阶：
你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？

链接：https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array
"""
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 找开头
        target_start = -1
        start, end = 0, len(nums) - 1
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] < target:
                start += 1
            elif nums[mid] > target:
                end -= 1
            elif mid > 0 and nums[mid - 1] == target:
                end -= 1
            else:
                target_start = mid
                break

        # 找结尾
        target_end = -1
        start, end = 0, len(nums) - 1
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] < target:
                start += 1
            elif nums[mid] > target:
                end -= 1
            elif mid < len(nums) - 1 and nums[mid + 1] == target:
                start += 1
            else:
                target_end = mid
                break
        
        return [target_start, target_end]
```



### 55. Jump game

核心方法：`贪心`

解题思路：

```python
"""
给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

链接：https://leetcode-cn.com/problems/jump-game/
"""
```



### 45. Jump game II

核心方法：`贪心`

解题思路：

```python
"""
给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

假设你总是可以到达数组的最后一个位置。

链接：https://leetcode-cn.com/problems/jump-game-ii
"""
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



### 74. Search a 2D matrix

核心方法：`二分`

解题思路：将二分的中点与矩阵中的坐标做映射

```python
"""
编写一个高效的算法来判断m x n矩阵中，是否存在一个目标值。该矩阵具有如下特性：

每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。

链接：https://leetcode-cn.com/problems/search-a-2d-matrix
"""
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows = len(matrix)
        cols = len(matrix[0])
        num_count = rows * cols

        start, end = 0, num_count - 1
        while start <= end:
            mid = start + (end - start) // 2
            mid_row = mid // cols
            mid_col = mid % cols

            if matrix[mid_row][mid_col] < target:
                start += 1
            elif matrix[mid_row][mid_col] > target:
                end -= 1
            else:
                return True
        
        return False
```



### 75. Sort colors

核心方法：`双指针`

解题思路：类似快排中的partition算法

```python
"""
给定一个包含红色、白色和蓝色，一共n个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数0、1和2分别表示红色、白色和蓝色。

链接：https://leetcode-cn.com/problems/sort-colors
"""
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        # 数组分为3个区域，“0区”、“1区”和“2区”，两个指针分别指向“1区”第一个元素和“2区”第一个元素
        # [00..0  11..1   22...2  xxx.xxx]
        #         |       |       |
        #         p0_next p1_next i
        p0_next, p1_next = 0, 0
        # i指向未知区域第一个元素
        for i in range(len(nums)):
            if nums[i] == 0:
                nums[i], nums[p0_next] = nums[p0_next], nums[i]
                if p1_next > p0_next:
                    nums[i], nums[p1_next] = nums[p1_next], nums[i]
                p0_next += 1
                p1_next += 1
            elif nums[i] == 1:
                nums[i], nums[p1_next] = nums[p1_next], nums[i]
                p1_next += 1
            # i指向2直接往后走
```



快排partition模板

```python
```



全组合/全排列问题运用了 __回溯__ 的思想，

- 回溯本质是一个 __遍历__ 问题，相比完全暴力，回溯的优化点在于增量式修改，即每次都从已有结果出发进行局部探索
- 既然是遍历，使用 __DFS__ 或 __BFS__ 都可以，分别对应栈和队列来存储组合

### 78. Subsets

核心方法：`回溯`

解题思路：属于经典的 __全组合__ 问题

- __队列__ 存储已有组合（空组合时初始元素）
- 依次枚举每个新元素，并选择 追加 或 不追加 到各已有组合的结尾
- 枚举结束时队列中保存了所有组合

```python
"""
给你一个整数数组nums，数组中的元素互不相同。返回该数组所有可能的子集（幂集）。
解集不能包含重复的子集。你可以按任意顺序返回解集。

链接：https://leetcode-cn.com/problems/subsets/
"""
from collections import deque
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        if not nums:
            return []

        queue = deque()
        # 初始元素是什么都不选的空组合
        queue.append([])
		# 依次枚举每个新元素
        for num in nums:
            existed_combo_count = len(queue)
            for _ in range(existed_combo_count):
                existed_combo = queue.popleft()
                # 每一个新元素，选择 追加 或 不追加 到已有组合结尾
                queue.append(existed_combo + [])
                queue.append(existed_combo + [num])

        return list(queue)
```



### 90. Subsets II

核心方法：`回溯`

解题思路：

```python
"""
给你一个整数数组nums，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。
解集不能包含重复的子集。返回的解集中，子集可以按任意顺序排列。

链接：https://leetcode-cn.com/problems/subsets-ii
"""

```



### 39. Combination sum

核心方法：`回溯`

解题思路：本题使用递归，因此本质上是基于 __栈__ 的回溯

```python
"""
给定一个无重复元素的数组candidates和一个目标数target，找出candidates中所有可以使数字和为target的组合。
candidates中的数字可以无限制重复被选取。

说明：
所有数字（包括target）都是正整数。
解集不能包含重复的组合。 

链接：https://leetcode-cn.com/problems/combination-sum
"""
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        self._recur(candidates, target, [], result)

        return result
    
    def _recur(self, candidates: List[int], target: int, curr: List[int], result: List[List[int]]):
        if target == 0:
            result.append(list(curr))
            return

        for i in range(len(candidates)):
            num = candidates[i]
            curr.append(num)

            if target >= num:
                # 在当前num之前的candidates不必遍历，因为如果可行，已经在之前的遍历中遇到过了
                self._recur(candidates[i:], target - num, curr, result)

            curr.pop()
```



### 40. Combination sum II

核心方法：

解题思路：

```python
"""
给定一个数组candidates和一个目标数target，找出candidates中所有可以使数字和为target的组合。

candidates中的每个数字在每个组合中只能使用一次。

说明：
所有数字（包括目标数）都是正整数。
解集不能包含重复的组合。 

链接：https://leetcode-cn.com/problems/combination-sum-ii
"""
```



### 216. Combination sum III

核心方法：`DFS`，`递归遍历`

解题思路：

```python
"""
找出所有相加之和为n的k个数的组合。组合中只允许含有1 - 9的正整数，并且每种组合中不存在重复的数字。

说明：
所有数字都是正整数。
解集不能包含重复的组合。 

链接：https://leetcode-cn.com/problems/combination-sum-iii
"""
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        result = []
        self._recur(n, k, 1, [], result)
        return result
    
    def _recur(self, target: int, k: int, i: int, curr: List[int], result: List[List[int]]):
        if k == 0:
            if target == 0:
                result.append(list(curr))
            return

        for num in range(i, 10):
            curr.append(num)
            if target >= num:
                self._recur(target - num, k - 1, num + 1, curr, result)
            curr.pop()
```



### 46. Permutations

核心方法：

解决思路：

```python
"""
给定一个不含重复数字的数组nums，返回其所有可能的全排列。你可以按任意顺序返回答案。

链接：https://leetcode-cn.com/problems/permutations/
"""
from collections import deque

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        queue = deque()
        queue.append([])

        for num in nums:
            existed_combo_count = len(queue)
            for _ in range(existed_combo_count):
                existed_combo = queue.popleft()
                for i in range(len(existed_combo) + 1):
                    if i == 0:
                        queue.append(list([num] + existed_combo))
                    elif i == len(existed_combo):
                        queue.append(list(existed_combo + [num]))
                    else:
                        queue.append(list(existed_combo[:i] + [num] + existed_combo[i:]))
        
        return list(queue)
```



### 47. Permutations II

核心方法：

解题思路：

```python
"""
给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。

链接：https://leetcode-cn.com/problems/permutations-ii/
"""

https://leetcode-cn.com/problems/combination-sum/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-m-2/
```

---



回溯：

DFS注意状态的回退，BFS则没有这一问题

### 79. Word search

核心方法：`回溯`

解题思路：

```python
"""
给定一个m x n二维字符网格board和一个字符串单词word。如果word存在于网格中，返回true；否则，返回false。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

链接：https://leetcode-cn.com/problems/word-search
"""
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board:
            return False
        
        row_count = len(board)
        col_count = len(board[0])
        for i in range(row_count):
            for j in range(col_count):
                # 状态矩阵置零
                status = []
                for _ in range(row_count):
                    status.append([0] * col_count)
                # 以i, j为起点搜索
                if self.match(board, status, i, j, word):
                    return True
        return False
    
    def match(self, board, status, start_i, start_j, word):
        # 整个单词都遍历完了
        if len(word) == 0:
            return True
        # 超出边界
        if not (0 <= start_i < len(board) and 0 <= start_j < len(board[0])):
            return False
        # 已访问过的位置
        if status[start_i][start_j] == 1:
            return False
        # 不符合要求的路径
        if board[start_i][start_j] != word[0]:
            return False

        status[start_i][start_j] = 1

        # 沿上下左右四个方向搜索
        if self.match(board, status, start_i - 1, start_j, word[1:]):
            return True
        if self.match(board, status, start_i + 1, start_j, word[1:]):
            return True
        if self.match(board, status, start_i, start_j - 1, word[1:]):
            return True
        if self.match(board, status, start_i, start_j + 1, word[1:]):
            return True
        
        # 注意：回溯前状态的清除
        status[start_i][start_j] = 0
        return False
```



### 80. Remove duplicates from sorted array II

核心方法：`双指针`，`partition`

解题思路：

```python
"""
给你一个有序数组nums，请你原地删除重复出现的元素，使每个元素最多出现两次，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在原地修改输入数组 并在使用O(1)额外空间的条件下完成。

链接：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii
"""
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # 双指针，一个指向当前数组，一个指向去重后的数组的末尾
        p_dedup, p = 1, 1
        dup_count = 1
        # 一共有三种情况：重复且次数<2，重复且次数>=2，不重复
        while p < len(nums):
            if nums[p] == nums[p_dedup - 1]:
                dup_count += 1
                # cond1
                if dup_count <= 2:
                    nums[p], nums[p_dedup] = nums[p_dedup], nums[p]
                    p_dedup += 1
                # cond2, do nothing
            # cond3
            else:
                dup_count = 1
                nums[p], nums[p_dedup] = nums[p_dedup], nums[p]
                p_dedup += 1
            
            p += 1
        
        return p_dedup
```



### 105. Construct binary tree from preorder and inorder traversal

核心方法：`二叉树`，`二分`

解题思路：

```python
"""
根据一棵树的前序遍历与中序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

链接：https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal
"""
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        """
        注意：任何一轮迭代中，preorder和inorder的长度都一样
        """
        if not preorder or not inorder:
            return None

        # 根据“前序遍历”确定当前root节点
        root = TreeNode(preorder[0])

        # 根据root节点将“中序遍历”划分为左右两部分
        i = 0
        while i < len(inorder) and inorder[i] != preorder[0]:
            i += 1
        left_inorder = inorder[0:i] if i > 0 else []
        right_inorder = inorder[i+1:] if i < len(inorder) - 1 else []

        # 根据“中序遍历”左半边元素数，决定剩余“前序遍历”的划分方式
        # left_inorder和left_preorder的长度始终保持一致，对right_inorder和right_preorder也同理
        left_preorder = preorder[1:(len(left_inorder)+1)] if left_inorder else []
        right_preorder = preorder[(len(left_inorder)+1):] if right_inorder else []
        root.left = self.buildTree(left_preorder, left_inorder)
        root.right = self.buildTree(right_preorder, right_inorder)

        return root
```



### 106. Construct binary tree from inorder and postorder traversal

核心方法：`二叉树`，`二分`

解题思路：

```python
"""
根据一棵树的中序遍历与后序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

链接：https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        """
        前序遍历和后序遍历用来查找顶点
        中序遍历用来查找左右子树
        区别：前序遍历从头部确定顶点，后序遍历从尾部确定顶点
        """
        if not inorder or not postorder:
            return None
        
        root = TreeNode(postorder[-1])

        i = 0
        while i < len(inorder):
            if inorder[i] == root.val:
                break
            i += 1
        left_inorder = inorder[:i] if i > 0 else []
        right_inorder = inorder[(i+1):] if i < len(inorder) - 1 else []

        # 任何一轮迭代中，inorder和postorder的长度都相同（因为要表示同一颗子树）
        right_postorder = postorder[-(len(right_inorder)+1):-1] if right_inorder else []
        left_postorder = postorder[:len(left_inorder)] if left_inorder else []

        root.left = self.buildTree(left_inorder, left_postorder)
        root.right = self.buildTree(right_inorder, right_postorder)

        return root
```



### 120. Triangle

核心方法：`贪心`

解题思路：

```python
"""
给定一个三角形 triangle ，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

链接：https://leetcode-cn.com/problems/triangle
"""
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # 用一个下三角阵保存状态
        # 从上到下遍历
        rows = len(triangle)
        for row in range(1, rows):
            cols = len(triangle[row])
            for col in range(cols):
                triangle[row][col] += min(
                    triangle[row - 1][col] if col < cols - 1 else float('inf'),
                    triangle[row - 1][col - 1] if col > 0 else float('inf'))
        
        return min(triangle[-1])
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



### 287. Find the duplicate number

核心方法：`双指针`

解题思路：

```python
"""
给定一个包含n + 1个整数的数组nums，其数字都在1到n之间（包括 1 和 n），可知至少存在一个重复的整数。

假设nums只有一个重复的整数，找出这个重复的数。

你设计的解决方案必须不修改数组nums且只用常量级O(1)的额外空间。

链接：https://leetcode-cn.com/problems/find-the-duplicate-number
"""
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow, fast = 0, 0
        while slow == 0 or slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]

        slow = 0
        while slow == 0 or slow != fast:
            slow = nums[slow]
            fast = nums[fast]

        return slow
```



