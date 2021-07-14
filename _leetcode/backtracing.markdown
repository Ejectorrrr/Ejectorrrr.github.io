---
layout: post
title: 回溯篇
category: 原创
---

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

