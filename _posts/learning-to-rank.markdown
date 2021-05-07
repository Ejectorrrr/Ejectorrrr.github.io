# learning to rank

列表排序的目标是对多个条目进行排序，这就意味着它的目标值是有结构的。与单值回归和单值分类相比，结构化目标要求解决两个被广泛提起的概念：

- 列表评价指标
- 列表训练算法



列表排序的评价指标

- 相关度
- 列表的整体得分

列表排序的评价指标经历了三个阶段：

- Precision and Recall
- Discounted Cumulative Gain(DCG)
- Expected Reciprocal Rank(ERR)



列表排序的训练，按照label来分：

- point-wise，回归或分类算法

- pair-wise，二分类，减少逆序对数量

- List-wise，直接优化nDCG、ERR



list rank是个检索问题，检索包含索引和查询两个过程。



参考：

[1]:https://tech.meituan.com/2018/12/20/head-in-l2r.html