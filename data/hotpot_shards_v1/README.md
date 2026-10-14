# Hotpot训练题池分片 v1

2026-10-14。原训练池90,447题，经结构检查、bridge题过滤、文本去重、排除已有700道训练候选及评测题/标准证据标题后，保留71,486题。固定seed20261014，按easy/medium/hard分别打乱，再轮流分配到10片。

10片的题目ID互不重叠；每片约7148—7150题。题目分片不等于文档、实体或关系模板完全隔离，不能保证SFT质量提高。

- `partition_indices/shard_01.jsonl`至`shard_10.jsonl`：所有分片的固定题目成员索引（原始行号、ID、难度），尚未为其余9片抽题建文档库。
- `shard_01/sample_300_v1/questions.jsonl`：只从第一片抽取的300道新训练题；easy60、medium120、hard120。没有按答案或模型表现选择。
- `shard_01/sample_300_v1/documents.jsonl`：上述300题全部context段落的并集，包含正确证据和干扰文档，内容相同才合并，共2979篇。不能换成旧训练库，也不能只索引标准证据。
- `retrieval_binding.json`位于样本目录：固定题目与文档SHA256及pool ID。每道题也有`retrieval_pool_id`。字段兼容现有questions/documents格式。
- `validation_report.json`：分片唯一性、样本归属、支持句索引、配套哈希、错用旧库拒绝及BM25查询检查均通过。

## 跑轨迹时必须使用的目录

```powershell
$dataDir = '.\data\hotpot_shards_v1\shard_01\sample_300_v1'
```

学生模型运行时，将这个目录传给`run_baselines.py --data-dir`；教师本地工具执行时，将这个目录传给`teacher_batch_session.py init --source-dir`。两条入口都检查绑定；样本文件或文档库哈希不匹配则拒绝执行。基线运行还会在结果目录与dataset快照中保存绑定，便于重放。

本次只建库，尚未生成轨迹、筛选SFT或训练。后续采样必须冻结该语料并记录文档哈希，不把“换了检索库”的变化误归因为策略学习。原有训练、开发与模型产物不覆盖。其余分片后续各抽300题时需使用新的独立样本目录及各自文档库。
