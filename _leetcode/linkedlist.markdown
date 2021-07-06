---
layout: post
title: 链表篇
---



```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### 2. Add two numbers

核心方法：

解题思路：

```python
"""
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

链接：https://leetcode-cn.com/problems/add-two-numbers
"""
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        inc = 0
        result = ListNode()
        p = result
        while l1 is not None or l2 is not None:
            if l1 is None:
                bit_sum = l2.val + inc
                l2 = l2.next
            elif l2 is None:
                bit_sum = l1.val + inc
                l1 = l1.next
            else:
                bit_sum = l1.val + l2.val + inc
                l1 = l1.next
                l2 = l2.next
            inc = bit_sum // 10
            p.next = ListNode(bit_sum % 10)
            p = p.next

        if inc > 0 :
            p.next = ListNode(inc)

        return result.next
```



### 19. Remove Nth node from end of list

核心方法：

解题思路：

```python
"""
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

进阶：你能尝试使用一趟扫描实现吗？

链接：https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
"""
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        
```



### 24. Swap nodes in pairs

核心方法：

解题思路：

```python
"""
给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

链接：https://leetcode-cn.com/problems/swap-nodes-in-pairs/
"""
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head

        pre_node = head
        cur_node = pre_node.next
        suf_node = cur_node.next

        cur_node.next = pre_node
        pre_node.next = self.swapPairs(suf_node)

        return cur_node
```



### 61. Rotate list

核心方法：

解题思路：

```python
"""
给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。

链接：https://leetcode-cn.com/problems/rotate-list/
"""
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if head is None or head.next is None or k == 0:
            return head

        list_len = 0
        p = head
        while p is not None:
            list_len += 1
            p = p.next

        first, second = head, head
        while k % list_len > 0 and second is not None:
            second = second.next
            k -= 1

        # 将second移动到尾节点，first将指向待旋转部分的前一节点
        while second.next is not None:
            first = first.next
            second = second.next
        
        second.next = head
        new_head = first.next
        first.next = None

        return new_head
```



### 82. Remove duplicates from sorted list II

核心方法：

解题思路：

```python
"""
存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中 没有重复出现 的数字。

返回同样按升序排列的结果链表。

链接：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii
"""
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # 想清楚到底是三指针还是二指针
        new_head = ListNode(next=head)
        pre, cur, suf = new_head, head, head

        # cur指向重复区间首元素，suf指向重复区间尾元素的下一元素
        while pre is not None:
            # 判断[cur, suf)区间是否重复
            rep_count = 1
            while suf is not None and (suf == cur or suf.val == cur.val):
                if suf != cur:
                    rep_count += 1
                suf = suf.next

            # 以cur开头的区间不重复
            if rep_count == 1:
                pre.next = cur
                pre = cur
            # 无论以cur开头的区间是否重复，cur都要更新
            cur = suf

        return new_head.next
```



### 86. Partition list

核心方法：

解题思路：链表版partition函数

```python
"""
给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。

你应当 保留 两个分区中每个节点的初始相对位置。

链接：https://leetcode-cn.com/problems/partition-list
"""
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # 双指针，一个指向<=x的链表，一个指向>x的链表
        less_head = ListNode()
        more_head = ListNode()

        p_less = less_head
        p_more = more_head
        p = head
        while p is not None:
            if p.val < x:
                p_less.next = p
                p_less = p_less.next
            else:
                p_more.next = p
                p_more = p_more.next
            p = p.next
        
        p_more.next = None
        p_less.next = more_head.next

        return less_head.next
```



### 92. Reverse linked list II

核心方法：

解题思路：

```python
"""
给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。

链接：https://leetcode-cn.com/problems/reverse-linked-list-ii
"""
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        if head is None or head.next is None or left == right:
            return head
        
        lpre, lcur = None, head
        i = 1
        while lcur is not None and i < left:
            lpre = lcur
            lcur = lcur.next
            i += 1
        if lcur is None:
            return head
        
        rcur = lcur
        i = left
        while rcur is not None and i < right:
            rcur = rcur.next
            i += 1
        if rcur is None:
            return head
        
        # 从首元素开始反转
        if lpre is None:
            return self.reverse(rcur.next, lcur, rcur)
        else:
            lpre.next = self.reverse(rcur.next, lcur, rcur)
            return head
    
    def reverse(self, left_pre: ListNode, left: ListNode, right: ListNode) -> ListNode:
        if left == right:
            left.next = left_pre
            return left
        
        new_tail = left.next
        new_head = self.reverse(left, left.next, right)
        left.next = left_pre

        return new_head
```



### 109. Convert sorted list to binary search tree

核心方法：

解题思路：

```python
"""
给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。

链接：https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree
"""
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        list_len = self.length(head)
        if list_len == 0:
            return None
        if list_len == 1:
            return TreeNode(head.val)
        if list_len == 2:
            return TreeNode(head.next.val, TreeNode(head.val))
        
        # 大于等于3个节点
        pre, cur = None, head
        for _ in range((list_len - 1) // 2):
            pre = cur
            cur = cur.next
        # 断开前一段
        pre.next = None
        left_head = head
        right_head = cur.next

        root = TreeNode(val=cur.val, 
                        left=self.sortedListToBST(left_head), 
                        right=self.sortedListToBST(right_head))

        return root
    
    def length(self, head: ListNode) -> int:
        count = 0
        p = head
        while p is not None:
            p = p.next
            count += 1
        return count
```



### 114. Flatten binary tree to linked list

核心方法：

解题思路：

```python
"""
给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。

链接：https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list
"""
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        实际应该使用后序遍历
        """
        self.helper(root)
    
    def helper(self, root: TreeNode) -> List[TreeNode]:
        """
        return: 反转后的头节点和尾节点，都是TreeNode类型
        """
        if root is None:
            return None
        
        # 直接把左右子树flatten
        if root.left is not None and root.right is not None:
            left_head, left_tail = self.helper(root.left)
            right_head, right_tail = self.helper(root.right)
            root.right = left_head
            root.left = None
            left_tail.right = right_head
            return [root, right_tail]
        elif root.left is not None:
            head, tail = self.helper(root.left)
            root.right = head
            root.left = None
            return [root, tail]
        elif root.right is not None:
            head, tail = self.helper(root.right)
            root.right = head
            root.left = None
            return [root, tail]
        else:
            return [root, root]
```



### 138. Copy list with random pointer

核心方法：

解题思路：

```python
"""
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。

例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。

返回复制链表的头节点。

用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：

val：一个表示 Node.val 的整数。
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
你的代码 只 接受原链表的头节点 head 作为传入参数。

链接：https://leetcode-cn.com/problems/copy-list-with-random-pointer
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if head is None:
            return None

        # 穿插构造
        p = head
        while p is not None:
            p_copy = Node(x=p.val, next=p.next)
            p.next = p_copy
            p = p.next.next
        
        # 设置random pointer
        p = head
        while p is not None:
            if p.random is not None:
                p.next.random = p.random.next
            p = p.next.next
        
        # 穿插分裂
        old = head
        new = head.next
        res = head.next
        while old is not None:
            old.next = old.next.next
            if old.next is None:
                new.next = None
            else:
                new.next = new.next.next
            old = old.next
            new = new.next
        
        return res
```



### 142. Linked list cycle II

核心方法：

解题思路：

```python
"""
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。

说明：不允许修改给定的链表。

进阶：
你是否可以使用 O(1) 空间解决此题？

链接：https://leetcode-cn.com/problems/linked-list-cycle-ii
"""
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if head is None:
            return None

        # 找到相遇点
        slow, fast = head, head.next
        while fast is not None and fast.next is not None and slow != fast:
            slow = slow.next
            fast = fast.next.next
        
        if fast is None or fast.next is None:
            return None
        
        start = ListNode(next=head)
        while start != slow:
            start = start.next
            slow = slow.next
        
        return start
```



### 143. Reorder list

核心方法：

解题思路：

```python
"""
给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

链接：https://leetcode-cn.com/problems/reorder-list
"""
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # DFS, 栈

        list_len = self.length(head)
        if list_len < 3:
            return head
        
        self.recur_reorder(head, 0, list_len)

    def recur_reorder(self, head, i, total_len) -> ListNode:
        """
        返回当前reorder区间的下一节点，也就是其父区间的右节点
        """
        if i == (total_len - 1) // 2:
            if total_len % 2 == 0:
                cur_left, cur_right = head, head.next
            else:
                cur_left = cur_right = head
            parent_right = cur_right.next
            cur_right.next = None
            return parent_right
        
        cur_left, suf_left = head, head.next
        cur_right = self.recur_reorder(head.next, i + 1, total_len)
        cur_left.next = cur_right
        parent_right = cur_right.next
        cur_right.next = suf_left

        return parent_right
    
    def length(self, head: ListNode) -> int:
        count = 0
        p = head
        while p is not None:
            count += 1
            p = p.next
        return count
```



### 146. LRU cache

核心方法：

解题思路：

```python
"""
运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制 。
实现 LRUCache 类：

LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。

链接：https://leetcode-cn.com/problems/lru-cache
"""
class DLinkedList:
    def __init__(self, val, pre, nxt):
        self.val = val
        self.pre = pre
        self.nxt = nxt

class LRUCache:
    """
    双向链表维护访问时效（使用双向链表是为了O(1)时间移动节点）
    哈希表维护key到链表中节点的映射
    """

    def __init__(self, capacity: int):
        # 容量
        self.max_capacity = capacity
        self.cur_capacity = 0
        # 哈希表
        self.container = {}
        # 双向链表
        self.head = None
        self.tail = None

    def get(self, key: int) -> int:
        if key in self.container:
            self.move_to_end(key)
            return self.container[key].val[1]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.container:
            # 更新
            self.container[key].val = (key, value)
            # 冒泡
            self.move_to_end(key)
        else:
            # 尾部插入
            self.container[key] = DLinkedList((key, value), None, None)
            self.cur_capacity += 1
            if self.cur_capacity == 1:
                self.head = self.tail = self.container[key]
            else:
                self.tail.nxt = self.container[key]
                self.tail.nxt.pre = self.tail
                self.tail = self.tail.nxt
            # 删除头部
            while self.cur_capacity > self.max_capacity:
                head_key = self.head.val[0]
                del self.container[head_key]
                head_next = self.head.nxt
                del self.head
                self.head = head_next
                self.cur_capacity -= 1

    def move_to_end(self, key):
        if key not in self.container or self.cur_capacity == 1 or self.container[key] == self.tail:
            return

        cur_node = self.container[key]
        # 区分待移动节点是否是头节点
        if cur_node == self.head:
            self.head = cur_node.nxt
        else:
            cur_node.pre.nxt = cur_node.nxt
        cur_node.nxt.pre = cur_node.pre
        # 移动
        cur_node.pre = self.tail
        cur_node.nxt = None
        self.tail.nxt = cur_node
        self.tail = cur_node

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```



### 147. Insertion sort list

核心方法：

解题思路：

```python
"""
对链表进行插入排序。

插入排序的动画演示如上。从第一个元素开始，该链表可以被认为已经部分排序（用黑色表示）。
每次迭代时，从输入数据中移除一个元素（用红色表示），并原地将其插入到已排好序的链表中。

插入排序算法：
插入排序是迭代的，每次只移动一个元素，直到所有元素可以形成一个有序的输出列表。
每次迭代中，插入排序只从输入数据中移除一个待排序的元素，找到它在序列中适当的位置，并将其插入。
重复直到所有输入数据插入完为止。

链接：https://leetcode-cn.com/problems/insertion-sort-list
"""
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        sorted_head, cur = ListNode(next=None), head
        while cur is not None:
            # 把链表分为三个区域
            # sorted, cur, cur.next
            # 分别是已排序的、待插入的、未探索的

            # 1.取出：将cur从当前链表中取出
            suf = cur.next
            cur.next = None

            # 2.查找：在sorted区域的插入位置，即第一个不小于待插入节点的位置
            p = sorted_head
            while p.next is not None and p.next.val < cur.val:
                p = p.next
            
            # 3.插入：位置就在p和p.next之间
            cur.next = p.next
            p.next = cur

            cur = suf
        
        return sorted_head.next
```



### 148. Sort list

核心方法：

解题思路：

```python
"""
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

进阶：
你可以在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？

链接：https://leetcode-cn.com/problems/sort-list
"""
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # 二分？分治？
        list_len = self.length(head)
        return self.sortPart(head, list_len)

    def sortPart(self, head: ListNode, length: int) -> ListNode:
        if length < 2:
            return head

        # 找到分裂点，p是left的尾节点
        count = 0
        p = head
        while count < (length - 1) // 2:
            count += 1
            p = p.next
        left_head = head
        right_head = p.next
        p.next = None

        sorted_left = self.sortPart(left_head, (length + 1) // 2)
        sorted_right = self.sortPart(right_head, length - (length + 1) // 2)
        return self.mergeTwoSorted(sorted_left, sorted_right)

    def mergeTwoSorted(self, head1: ListNode, head2: ListNode) -> ListNode:
        if head1 is None:
            return head2
        if head2 is None:
            return head1
        
        if head1.val < head2.val:
            head1.next = self.mergeTwoSorted(head1.next, head2)
            return head1
        else:
            head2.next = self.mergeTwoSorted(head1, head2.next)
            return head2
    
    def length(self, head: ListNode) -> int:
        p = head
        count = 0
        while p is not None:
            p = p.next
            count += 1
        return count
```



### 328. Odd even linked list

核心方法：

解题思路：

```python
"""
给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。

请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。

链接：https://leetcode-cn.com/problems/odd-even-linked-list
"""
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        # 递归
        if head is None or head.next is None:
            return head

        odd_head, even_head = head, head.next
        pre, cur, suf = None, odd_head, even_head
        if_odd = 1
        while suf is not None:
            cur.next = suf.next
            pre = cur
            cur = suf
            suf = suf.next
            if_odd = if_odd ^ 1
        # 当链表长度为奇数时，cur指向奇链表尾；否则，cur指向偶链表尾
        if if_odd == 1:
            cur.next = even_head
        else:
            pre.next = even_head

        return odd_head
```



### 382. Linked list random node

核心方法：

解题思路：

```python
"""
给定一个单链表，随机选择链表的一个节点，并返回相应的节点值。保证每个节点被选的概率一样。

进阶:
如果链表十分大且长度未知，如何解决这个问题？你能否使用常数级空间复杂度实现？

链接：https://leetcode-cn.com/problems/linked-list-random-node
"""
class Solution:
    # 依然是蓄水池算法
    # 以1/n的概率保留遇到的第n个数即可
    def __init__(self, head: ListNode):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        """
        self.head = head

    def getRandom(self) -> int:
        """
        Returns a random node's value.
        """
        n = 0
        head = self.head
        result = 0
        while head is not None:
            n += 1
            if random.randrange(n) == 0:
                result = head.val
            head = head.next
        return result

# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()
```



### 445. Add two numbers II

核心方法：

解题思路：

```python
"""
给你两个 非空 链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。

你可以假设除了数字 0 之外，这两个数字都不会以零开头。

进阶：
如果输入链表不能修改该如何处理？换句话说，你不能对列表中的节点进行翻转。

链接：https://leetcode-cn.com/problems/add-two-numbers-ii
"""
from typing import Tuple
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 补齐两个链表的长度
        l1_len = self.length(l1)
        l2_len = self.length(l2)
        if l2_len < l1_len:
            l1, l2 = l2, l1
            l1_len, l2_len = l2_len, l1_len
        
        for _ in range(l2_len - l1_len):
            tmp = ListNode(0)
            tmp.next = l1
            l1 = tmp
        
        inc = self.recur_add(l1, l2, 0)
        if inc > 0:
            tmp = ListNode(inc)
            tmp.next = l1
            l1 = tmp
        return l1
    
    def recur_add(self, l1: ListNode, l2: ListNode, inc: int) -> int:
        if l1 is None and l2 is None:
            return 0
        
        inc = self.recur_add(l1.next, l2.next, inc)
        sum_ = l1.val + l2.val + inc
        l1.val = sum_ % 10
        inc = sum_ // 10
        return inc

    def length(self, head: ListNode) -> int:
        p = head
        count = 0
        while p is not None:
            p = p.next
            count += 1
        return count
```



### 707. Design linked list

核心方法：

解题思路：

```python
"""
设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：val 和 next。val 是当前节点的值，next 是指向下一个节点的指针/引用。如果要使用双向链表，则还需要一个属性 prev 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。

在链表类中实现这些功能：

get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。
deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。

链接：https://leetcode-cn.com/problems/design-linked-list
"""
class MyLinkedList:

    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.size = 0
        self.head = None


    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if 0 <= index < self.size:
            p = self.head
            while p is not None and index > 0:
                p = p.next
                index -= 1
            return p.val
        else:
            return -1


    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        self.head = ListNode(val=val, next=self.head)
        self.size += 1


    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        if self.head is None:
            self.addAtHead(val)
        else:
            p = self.head
            while p.next is not None:
                p = p.next
            p.next = ListNode(val=val)
            self.size += 1


    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index == 0:
            self.addAtHead(val)
        elif index == self.size:
            self.addAtTail(val)
        elif 0 < index < self.size:
            pre, cur = None, self.head
            while cur is not None and index > 0:
                pre = cur
                cur = cur.next
                index -= 1
            pre.next = ListNode(val=val, next=cur)
            self.size += 1


    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if 0 <= index < self.size:
            pre, cur = None, self.head
            while cur is not None and index > 0:
                pre = cur
                cur = cur.next
                index -= 1
            if pre is None:
                self.head = cur.next
                cur.next = None
            else:
                pre.next = cur.next
                cur.next = None
            del cur
            self.size -= 1

# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
```



### 725. Split linked list in parts

核心方法：

解题思路：

```python
"""
给定一个头结点为 root 的链表, 编写一个函数以将链表分隔为 k 个连续的部分。

每部分的长度应该尽可能的相等: 任意两部分的长度差距不能超过 1，也就是说可能有些部分为 null。

这k个部分应该按照在链表中出现的顺序进行输出，并且排在前面的部分的长度应该大于或等于后面的长度。

返回一个符合上述规则的链表的列表。

举例： 1->2->3->4, k = 5 // 5 结果 [ [1], [2], [3], [4], null ]

链接：https://leetcode-cn.com/problems/split-linked-list-in-parts
"""
class Solution:
    def splitListToParts(self, root: ListNode, k: int) -> List[ListNode]:
        # 推断：每段的节点个数最多包含两种选择
        # 设每段的节点数最多为k，共有a段包含k个节点，则
        # x * a + (x - 1) * (k - a) = size
        # a + (x - 1) * k = size, a <= k, (size - a) % k  == 0
        # 待搜索的变量主要是a

        size = self.length(root)
        a, x = None, None
        if size <= k:
            a = size
            x = 1
        else:
            # 根据上述条件搜索a的合理值
            for i in range(1, k + 1):
                if (size - i) % k == 0:
                    a = i
                    x = (size - a) // k + 1
                    break

        # 前a段每段有x个节点，后k-a段每段有x-1个节点
        results = []
        p = ListNode(next=root)
        for i in range(1, k + 1):
            results.append(p.next)
            # 按x分割
            if i <= a:
                for _ in range(x):
                    p = p.next
            # 按x-1分割
            else:
                for _ in range(x - 1):
                    p = p.next
            p_next = p.next
            p.next = None
            p = ListNode(next=p_next)

        return results

    def length(self, head: ListNode) -> int:
        count = 0
        p = head
        while p is not None:
            count += 1
            p = p.next
        return count
```



### 1669. Merge in between linked lists

核心方法：

解题思路：

```python
"""
给你两个链表 list1 和 list2 ，它们包含的元素分别为 n 个和 m 个。

请你将 list1 中第 a 个节点到第 b 个节点删除，并将list2 接在被删除节点的位置。

链接：https://leetcode-cn.com/problems/merge-in-between-linked-lists
"""
class Solution:
    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
        pre, cur = ListNode(next=list1), list1
        count = 1
        while cur is not None and count <= a:
            pre = cur
            cur = cur.next
            count += 1
        pre.next = list2

        while cur is not None and count <= b:
            pre = cur
            cur = cur.next
            count += 1
        
        p = list2
        while p.next is not None:
            p = p.next
        p.next = cur.next
        cur.next = None
        
        return list1
```

