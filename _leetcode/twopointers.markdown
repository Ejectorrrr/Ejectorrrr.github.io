---
layout: post
title: 双指针篇
---



## 最大/最小

核心方法：`双指针`/`滑动窗口`

解题思路：此类问题的最终解都出现在`一个`确定位置（一个窗口），考虑的重点是 __1)__ 左右边界从何处开始，__2)__ 如何移动两边界以到达目标位置。

---

### 3. Longest substring without repeating characters

- 初始：左右边界都从左侧开始
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



### 209. Minimum size subarray sum

- 初始：左右边界都从左侧开始
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



### 159. Longest substring with at most two distinct characters

- 初始：左右边界都从左侧开始

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



### 424. Longest Repeating Character Replacement

- 初始：左右边界都从左侧开始

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



### 76. Minimum window substring

- 初始：左右边界都从左侧开始

- 移动：右边界无条件移动，左边界当“窗口内涵盖了t中所有字符”时移动

__注__：当求最小时（本题, 209），左边界都是在`满足要求`时移动；而求最大时（3, 159, 424），左边界都是在`不满足要求`时移动

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

#### 模板：

```python
start, end = 0, 0
for end in range(len(arr)):
    ## some update operation
    while "shrink condition" and start <= end:
        ## some shrink operation
        start += 1 # shrink start
    # extend end without condition
# return result
```

---



## 非最大/最小

此类问题的解不是一个固定的位置，而是有很多个