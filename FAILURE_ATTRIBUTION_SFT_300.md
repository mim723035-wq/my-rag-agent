# 300条SFT：12道退步题归因

范围：同一100题开发集里，100条模型EM匹配、300条模型EM不匹配的12题。不是全部错误，也不完整解释净EM变化（另有7道新增匹配）。官方评分、题目、模型和训练数据未修改。当前助手阅读两版轨迹及固定文档证据，不是独立人工审定。

| 互斥主因 | 数量 | 题目 |
|---|---:|---|
| 查询未沿关键关系推进 | 3 | Craig Virgin挑战者出生地；Mineola词义；Elgar配音者与漫画 |
| 关键约束未确认就停止 | 3 | Schottenstein出生1954；Roger Groot大学所在县；Streak另一演员出生年 |
| 已有证据但关系判断错误 | 2 | 两部电影共同演员；两个人共同职业 |
| 长引用ID错误且恢复失败 | 1 | Casino Royale小说作者 |
| 证据支持的表述差异 | 2 | Flint, Michigan；Topeka, Kansas |
| 问题与标准证据不一致 | 1 | Orville Siegfried大学/高中混淆 |

合计12：8题语义/决策错误、1题引用执行失败、2题表述差异、1题题目质量问题。主因分类用于避免重复计数，次要因素可以重叠。例如早停题也有实体选择或答案类型混淆。

## 逐题证据

1. **Craig Virgin题**（5a7a3cea5542994f819ef182）：首篇明确列出Chris Derrick与Lukas Verzbicas为接近纪录者。新版第二查仍围绕Craig，回答其出生地Belleville。旧版查Lukas出生地得到Kaunas。主因为中间实体切换失败，不是首跳未找到。
2. **Casino Royale题**（5a86f2fb554299211dda2b64）：新版实际反复生成正确答案Ian Fleming；一个长ID未被返回，5次finish遭拒，末次JSON被截断，最终token_limit。不能把最终空答案解释为不懂作者；主要是引用复制和错误反馈恢复。
3. **Baby Driver题**（5a87b30c5542994846c1cdcf）：第二查已返回两部电影演员表，共同演员Kevin Spacey可直接验证，但回答Ansel Elgort。检索充分，错误位于集合交集/实体约束判断。
4. **共同职业题**（5a886098554299206df2b248）：Alexandra文档支持writer；Kerouac文档支持novelist and poet。回答novelist and poet未证明两者都为诗人，不能像地名省略一样直接接受为同义答案。
5. **Schottenstein题**（5a8b98665542995d1e6f1413）：首查列出Jerome与Jay，但没有1954出生证据；新版直接选择Jerome。旧版再查取得Jay出生1954。主因为约束未核验即停止。
6. **Flint题**（5abaf0ec5542996cc5e49f22）：两版搜索及引用一致，Flint, Michigan只是省略United States，全文支持。官方EM仍保留不匹配，人工归因标为表述差异。
7. **Mineola题**（5abd8ee05542993062266cc8）：首查选区列出许多社区，新版先选Baldwin，第二查忽略pleasant place线索，未找到词义支持就提交。旧版虽也查询Baldwin，却保留词义词找到Mineola。主因为查询丢失关键约束，并有未经验证提交。
8. **Topeka题**（5abfa18f5542990832d3a176）：文档支持Topeka, Kansas，新版首查已取得完整亲属—教会—城市链并停止；旧版又查一次。是表述差异而非事实退步，且是合理提前停止的案例。
9. **Elgar题**（5ac3dfee554299204fd21eab）：首查已确定Derek Stephen Prince，后续却查询Power Rangers Turbo manga，没沿配音者—英语角色—Keitaro线索推进。Love Hina标准角色文档确实给出英语配音关系；新版选的另一漫画没有支持该关系。
10. **Roger Groot题**（5adfa73b55429942ec259ad8）：首查明确大学及Lexington，但未提供县；新版停止并以城市回答县。旧版查大学得到Rockbridge County。主因为停止过早，次因为答案类型未核对。
11. **Orville题**（5ae06ee755429945ae959352）：文档说高中在Lee's Summit、大学是Washington & Jefferson College，题目却问college所在城市，标准答案支持的是高中。旧版虽EM匹配，但也未证明大学城市关系；新版又以学院名回答城市。该题有错误前提，不用作干净模型退步证据，不更改原评分。
12. **Streak题**（5ae531a95542990ba0bbb1fc）：首查主演是Brittany Snow和Rumer Willis，问题要求与Rumer合演的人；新版直接用Rumer自身出生年1988。旧版查Brittany得到1986。主因为尚未确认目标实体及出生年就停止。

## 定向检索诊断

使用300条评测时的原文档库与同一BM25，额外执行诊断查询，不加入Agent原始轨迹或评分：

| 查询 | 第一名文档 |
|---|---|
| Lukas Verzbicas born | Lukas Verzbicas |
| Jay Schottenstein born 1954 | Jay Schottenstein |
| Brittany Snow born | Brittany Snow |
| Washington and Lee University county | Washington and Lee University |
| Nassau village pleasant place Kathleen Rice | Mineola, New York |
| Derek Stephen Prince Keitaro Arima | Keitarō Urashima |

这些正确目标文档可检索，支持“这6个案例不能只归咎于语料缺失或检索器完全不可用”。诊断查询由审核者设计，不代表模型自主生成能力，也不证明检索器在其他题没有问题。

## 可支持的结论

这批退步集中于**缺失关系没有被持续追踪、候选约束没有核验、已有证据没有求交集**。完整引用ID并不能保证引用支持所有关系。平均搜索减少同时包含合理提前停止（Topeka）和错误提前停止（大学县），应分开评价。

仍不能证明这些行为由新增数据比例、训练步数或过拟合直接导致；也不能凭12个案例改算整个100题的“语义准确率”。需要同等审阅新增7题及其他答案，并通过控制数据构成/训练预算的实验判断训练机制。后续可在新训练题中增加条件约束、多候选排除、缺失末跳继续搜索和引用修复示范；不得将本批开发题复制成训练数据。

来源：`outputs/baselines_20261013T162936Z_b0e435fe/agent`、`outputs/baselines_20261013T234622Z_601d10b9/agent`及其固定questions/documents快照。逐题带哈希审阅记录为新版`evaluation/failure_attribution_12.json`，实际查询结果为`evaluation/failure_targeted_retrieval_checks.json`。
