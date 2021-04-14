---
layout: post
title: 数组篇
---



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
    elif sum_ < target:
        start += 1
    else:
        end -= 1
return results
```

-----



### 4. Median of two sorted arrays

to be accepted



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
        left, right = 0, len(height) - 1
        while left < right:
            volume = max(volume, (right - left) * min(height[left], height[right]))
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

