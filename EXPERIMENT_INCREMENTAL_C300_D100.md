# 冻结C300，在其LoRA上增量训练新100条：C/D对照

2026-10-14。用户选择增量训练方案：保留C，加载C的可训练LoRA权重，仅学习新分片的100条轨迹一轮，另存D。不是从底座联合训练400条，也不是叠加第二套LoRA。

## 冻结与训练

- C原运行：`outputs/sft_expanded300_20261013T233348Z_306e9948`。
- C快照：`outputs/frozen_C300_v1`，所有适配器文件逐字节哈希一致；新训练和评测后原文件与快照均未改变。
- D训练：`outputs/sft_incremental100_20261014T044718Z_7d240a0d`。
- D适配器：该目录`adapter`。加载C后，逐张量核对与父检查点一致再更新；完整更新后的D权重与C不同。
- 新数据：`data/sft_targeted_shard_100_v2`，与旧300条父题ID无交集；257个动作、15,729监督token。v1草稿不用。
- 固定Qwen3-1.7B revision、BF16、q_proj/v_proj、r8/alpha16/dropout0、lr1e-4、batch1、梯度累积1、seed42、gradient checkpointing；新100条训练一轮。
- 257步，loss与梯度均有限；耗时260.53秒（约4.3分钟，不含加载），峰值已分配显存4.747GiB。
- 旧检查点未保存AdamW状态，因此重新初始化优化器。准确说是权重增量训练，不是恢复旧优化器状态的精确断点续训。
- D保存、重新加载及已见训练上下文推理通过，Peter O'Toole例的答案与引用均符合训练目标；仅作工程验证。

## 同预算开发集对照

C：`outputs/baselines_20261013T234622Z_601d10b9`。

D：`outputs/baselines_20261014T045233Z_5c5a6ab6`。

同一100题、同一文档库和BM25、初始提示词、底座revision与runtime逐题核验PASS，无缺失/损坏文件。关闭thinking、greedy，每题最多3次搜索、8个动作、768生成token、4096上下文。

| 指标 | C：300条 | D：C上再训100条 |
|---|---:|---:|
| 答案EM | 42% | 45% |
| 答案F1 | 0.5422 | 0.5910 |
| 完整标准文档引用率 | 56% | 56% |
| 标准证据检索召回 | 0.775 | 0.800 |
| 正常结束 | 99/100 | 100/100 |
| 平均搜索次数 | 1.77 | 1.99 |
| 平均重复搜索次数 | 0 | 0.07 |
| 平均输入token | 2526.00 | 2831.82 |
| 平均输出token | 180.09 | 175.67 |
| 未见引用ID错误事件 | 6 | 0 |
| 输出Cannot determine的题数 | 3 | 12 |

D新增10题EM匹配，同时失去7题，35题均匹配、48题均不匹配。EM净增3个百分点。5000次成对问题bootstrap（seed42）95%百分位区间[-5,+11]个百分点，跨0，不证明稳定显著改善。区间不覆盖训练随机种子波动。

引用错误事件减少是本次可见现象；Cannot determine增加也需关注。开发题在标注上均可回答，但弃答可能来自错误查询、漏召回或过度保守，不能仅凭数量断定受控缺证据训练造成保守。历史两版墙钟时延受系统负载影响，不作为可靠速度比较。

## 已核验案例

- **末跳继续查询改善**：Roger Groot大学所在县。C首查后以Lexington城市作答；D查到大学后继续查询`Washington Lee University county`，取得Rockbridge County并引用两段关系证据。
- **输出执行失败消失但重复搜索仍在**：Casino Royale作者。C反复使用未见长ID，最终token_limit；D提交Ian Fleming并引用实际见过的文档，但同一作者查询仍执行两次，不能说停止策略已完全修复。
- **退步**：巴西车手最多参赛纪录题。D首查明确Rubens Barrichello，却转而查询Emerson Fittipaldi，并重复该错误查询；最终Cannot determine。C回答11。该例是实体/查询推进失败，弃答不等于题目不可回答。
- 部分逐题EM变化是表述长度差异，如Topeka, Kansas→Topeka、racehorse breeder and owner→racehorse breeder、William Dwight Schultz→Dwight Schultz；未改官方评分，不把所有7道损失直接解释为事实错误。

## 结论与复验

该增量方案在本次开发集的EM/F1和正常结束率改善，引用ID错误事件降为0，但完整标准引用率未增加，搜索次数与重复搜索上升。保留C、D和此前B100，不能宣称D全面领先：B100的开发EM仍为47%，D为45%。

变化同时涉及新增数据的行为构成、训练顺序、257次更新、优化器重置及实际推理消耗；不能只归因为“见过400题”。未混入旧训练数据回放，出现的退步是可见现象，但不足以证明灾难性遗忘的机制。仍缺新盲测与多训练种子。

评分脚本：`scripts/evaluate_trace.py`；逐题配置核验与bootstrap：`scripts/compare_incremental_dev.py`。完整结果及变化列表位于D运行的`evaluation/comparison_300_incremental100.json`、`paired_answer_changes.json`。训练初始化、哈希与有限性证据见D训练目录`parent_initialization_check.json`、`training_verification.json`、`result.json`。
