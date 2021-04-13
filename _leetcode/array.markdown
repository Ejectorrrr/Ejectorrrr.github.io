---
layout: post
title: 数组篇
---



### 15. 3Sum

核心方法：`双指针夹逼`

解题思路：固定任意一个元素$O(N)$，其余两个元素按照双指针夹逼方式移动$O(N)$，总体是$O(N^2)$时间复杂度。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 排序：1. 便于跳过重复；2. 便于使用双指针
        sorted_nums = sorted(nums)

        results = []
        # 固定任意一个元素，其余两个元素按照双指针模式联动
        for first in range(len(sorted_nums) - 2):
            if first > 0 and sorted_nums[first] == sorted_nums[first - 1]:
                continue
            second = first + 1
            third = len(sorted_nums) - 1
            target = -sorted_nums[first]

            while second < third:
                if second > first + 1 and sorted_nums[second] == sorted_nums[second - 1]:
                    second += 1
                    continue
                if third < len(sorted_nums) - 1 and sorted_nums[third] == sorted_nums[third + 1]:
                    third -= 1
                    continue
                
                # 双指针模式
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

