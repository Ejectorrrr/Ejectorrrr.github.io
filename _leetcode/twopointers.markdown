---
layout: post
title: 双指针篇
category: 原创
---



方法体系：

- 双指针
  - 滑动窗口
  - 夹逼
- partition
- 快慢指针



## 一、双指针

### 1.1 最大/最小

核心方法：`滑动窗口`

解题思路：此类问题的最终解都出现在一个确定位置（或窗口）；思考的重点是，1) 左右边界从何处开始，2) 如何移动两边界以到达目标位置；特点是边界只能单向移动（从而保证$O(N)$时间复杂度）。



- 初始：左右边界都从左侧开始

#### 3. Longest substring without repeating characters

- 移动：右边界无条件移动，左边界当“发现重复”时移动

```python
"""
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

链接：https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/
"""
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dict_ = {}
        max_length = 0
        start = 0
        for end, ch in enumerate(s):
            # 更新窗口内的字符频率
            dict_[ch] = dict_.get(ch, 0) + 1
            # 因为去重策略的存在，只有新字符可能重复
            if dict_[ch] == 1:
                max_length = max(max_length, end - start + 1)
            else:
                # 去重
                while dict_[ch] > 1:
                    dict_[s[start]] -= 1
                    start += 1

        return max_length
```

#### 209. Minimum size subarray sum

- 移动：右边界无条件移动，左边界当“子数组和>=target”时移动

```python
"""
给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

链接：https://leetcode-cn.com/problems/minimum-size-subarray-sum
"""
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        start, end = 0, 0
        sub_sum = 0
        min_len = float('inf')
        for end, new_num in enumerate(nums):
            # 更新子数组和
            sub_sum += new_num
            # 满足要求时移动左边界
            while sub_sum >= target and start <= end:
                # 记录最小
                min_len = min(min_len, end - start + 1)
                sub_sum -= nums[start]
                start += 1

        return min_len if min_len < float('inf') else 0
```

#### 159. Longest substring with at most two distinct characters

- 移动：右边界无条件移动，左边界当”子数组内不同字符数>2“时移动

```python
"""
给定一个字符串 s ，找出 至多 包含两个不同字符的最长子串 t ，并返回该子串的长度。

链接：https://leetcode-cn.com/problems/longest-substring-with-at-most-two-distinct-characters/
"""
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        ch_count = {}
        start = 0
        longest = 0
        for end, ch in enumerate(s):
            # 更新窗口内不同字符数
            ch_count[ch] = ch_count.get(ch, 0) + 1
            # 记录最大
            if len(ch_count) <= 2:
                longest = max(longest, end - start + 1)
            # 不满足要求时移动左边界
            while len(ch_count) > 2 and start <= end:
                ch_count[s[start]] -= 1
                if ch_count[s[start]] == 0:
                    del ch_count[s[start]]
                start += 1

        return longest
```

#### 424. Longest Repeating Character Replacement

- 移动：右边界无条件移动，左边界当“所有非最高频字母的频率之和>k”时移动

```python
"""
给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，总共可最多替换 k 次。在执行上述操作后，找到包含重复字母的最长子串的长度。

链接：https://leetcode-cn.com/problems/longest-repeating-character-replacement
"""
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        start = 0
        ch_count = {}
        longest = 0
        for end, ch in enumerate(s):
            # 更新窗口内字频
            ch_count[ch] = ch_count.get(ch, 0) + 1
            # 记录最大
            replaced = sum(ch_count.values()) - max(ch_count.values())
            if replaced <= k:
                longest = max(longest, end - start + 1)
            # 不满足要求时移动左边界
            while replaced > k and start <= end:
                ch_count[s[start]] -= 1
                replaced = sum(ch_count.values()) - max(ch_count.values())
                start += 1

        return longest
```

#### 76. Minimum window substring

- 移动：右边界无条件移动，左边界当“窗口内涵盖了t中所有字符”时移动

__注__：当求 __最小__ 时（76, 209），左边界都是在`满足要求`时移动；而求 __最大__ 时（3, 159, 424），左边界都是在`不满足要求`时移动

```python
"""
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串。
注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。

链接：https://leetcode-cn.com/problems/minimum-window-substring
"""
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        t_map = {}
        for ch in t:
            t_map[ch] = t_map.get(ch, 0) + 1
        
        start = 0
        win_map = {}
        min_len, min_str = float('inf'), ''
        for end in range(len(s)):
            # 更新窗口字频
            win_map[s[end]] = win_map.get(s[end], 0) + 1
            # 满足要求时移动左边界
            while self._contain(win_map, t_map):
                # 记录最小
                if end - start + 1 < min_len:
                    min_len = end - start + 1
                    min_str = s[start:(end + 1)]
                win_map[s[start]] -= 1
                start += 1

        return min_str

    def _contain(self, map_a, map_b):
        for ch, count in map_b.items():
            if map_a.get(ch, 0) < count:
                return False
        return True
```

---

##### 模板：

```python
start, end = 0, 0
for end in range(len(arr)):
    ## some code for update operation
    while "shrink condition" and start <= end:
        ## some code for shrink operation
        start += 1 # shrink the start
    # extend the end anyway
# return result
```

---



- 初始：左右边界从两侧开始

#### 11. Container with most water

- 移动：在缩短宽度的同时应尽量提升高度，因此每次移动短板并更新新的容积

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

---

##### 模板

```python
start, end = 0, len(arr) - 1
while start < end:
    ## some code for update operation
    if "shrink condition":
    	start += 1
    else:
        end -= 1
# return result
```

---



### 1.2 非最大/最小

此类问题的解可能不唯一，典型的是2sum类问题。

核心方法：`夹逼`



#### 15. 3Sum

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

#### 16. 3Sum closest

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

#### 18. 4Sum

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

---

##### 2Sum模板：

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



## 二、partition

快排是分治+partition，而partition的核心思想是双指针。



##### 75. Sort colors

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

##### 80. Remove duplicates from sorted array II

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

---

partition模板

```python

```

---



## 三、快慢指针



##### 287. Find the duplicate number

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

