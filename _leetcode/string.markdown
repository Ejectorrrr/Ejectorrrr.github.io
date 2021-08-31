---
layout: post
title: 字符串篇
---

### 3. Longest Substring Without Repeating Characters

核心方法：

解题思路：

```python
"""
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
"""
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dict_ = {}
        max_length = 0
        start = 0
        for i, ch in enumerate(s):
            # 更新窗口内的字符频率
            if ch not in dict_:
                dict_[ch] = 0
            dict_[ch] += 1

            # 因为去重策略的存在，只有新字符可能重复
            if dict_[ch] == 1:
                max_length = max(max_length, i - start + 1)
            else:
                # 去重
                while dict_[ch] > 1:
                    dict_[s[start]] -= 1
                    start += 1

        return max_length
```



### 678. Valid parenthesis string

核心方法：`贪心`

解题思路：

```python
"""
给定一个只包含三种字符的字符串：（ ，） 和 *，写一个函数来检验这个字符串是否为有效字符串。有效字符串具有如下规则：

任何左括号 ( 必须有相应的右括号 )。
任何右括号 ) 必须有相应的左括号 ( 。
左括号 ( 必须在对应的右括号之前 )。
* 可以被视为单个右括号 ) ，或单个左括号 ( ，或一个空字符串。
一个空字符串也被视为有效字符串。

链接：https://leetcode-cn.com/problems/valid-parenthesis-string
"""
class Solution:
    def checkValidString(self, s: str) -> bool:
        # 待匹配的左括号数量为0时能够完成匹配
        # 在遍历过程中，待匹配的左括号数量有多种可能性，构成一个区间，我们记录其左右边界
        left, right = 0, 0
        for i in range(len(s)):
            if s[i] == '(':
                left += 1
                right += 1
            if s[i] == ')':
                left -= 1
                right -= 1
            if s[i] == '*':
                left -= 1
                right += 1
        return left <= 0 <= right
```

