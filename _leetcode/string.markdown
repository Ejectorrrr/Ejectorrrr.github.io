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

