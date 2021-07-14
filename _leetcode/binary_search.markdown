---
layout: post
title: 二分搜索篇
category: 原创
---



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



### 240. Search a 2D matrix II

核心方法：`二分`

解题思路：

```python
"""
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。

链接：https://leetcode-cn.com/problems/search-a-2d-matrix-ii
"""
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix:
            return False
        
        row_count = len(matrix)
        col_count = len(matrix[0])

        for row in range(row_count):
            if row == 0:
                end = col_count - 1
            else:
                end = start - 1
            start = 0

            while start <= end:
                mid = start + (end - start) // 2
                if matrix[row][mid] == target:
                    return True
                elif matrix[row][mid] < target:
                    start = mid + 1
                else:
                    end = mid - 1

        return False
```



### 378. Kth smallest element in a sorted matrix

核心方法：`二分`

解题思路：

```python
"""
给你一个 n x n 矩阵 matrix ，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
请注意，它是 排序后 的第 k 小元素，而不是第 k 个 不同 的元素。

链接：https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix
"""

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

