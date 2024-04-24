# GraduationProject
毕业设计

## DataSet数据集

[Replication Data for: A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling - MaiMemo Dataverse (harvard.edu)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VAGUL0)

`u` 学生id \
`w` 单词 \
`i` 用户复习的总次数，包含本次 \
`d` 单词的复杂度 \
`t_history` 历史时间间隔序列 \
`r_history` 回忆结果的序列 \
`delta_t` 上次复习的间隔 \
`r` 本次复习的结果 \
`p_recall` 复习成功的概率 \
`total_cnt` 相同序列的总数 

---

本文在半衰期回归（HLR）模型基础上提出了一个基于注意力机制的间隔重复模型Transformer-HLR，相较于目前表现最好的GRU-HLR模型，降低了大约3%的误差。我们同时也将注意力机制的引入到GRU-HLR模型，降低了大约2%的误差，并发现注意力机制有利于降低GRU-HLR在较大半衰期的预测误差。
