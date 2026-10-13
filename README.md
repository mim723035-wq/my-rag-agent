# 面向多跳问答的搜索 Agent 小规模 SFT

基于 Qwen3-1.7B、固定 HotpotQA 文档库和 BM25 检索，
实现不搜索、固定 RAG、模型自主搜索三种基线，
并使用审核后的搜索轨迹进行 LoRA SFT。

当前已完成：
- 基线、控制器、轨迹记录和批量评测。
- 41条审核轨迹、91个动作样本的LoRA训练。
- 100题开发评测、200题保留测试。
- 搜索次数上限1次与3次的开发集对照。
- 失败案例分析。

尚未进行RL或开放网页搜索泛化验证。

实验设置、结果和限制见：
[实验报告](EXPERIMENT_SFT_V1.md)

## 主要结果

200题保留测试，搜索上限均为3次：

| 指标 | 底座 | LoRA |
|---|---:|---:|
| 答案EM | 26.5% | 33.5% |
| 平均答案F1 | 0.3895 | 0.4605 |
| 完整标准文档引用率 | 4% | 16.5% |
| 正常结束 | 185/200 | 198/200 |
| 平均搜索次数 | 1.43 | 1.82 |
| 平均生成token | 133.71 | 153.54 |

100题开发集限制为最多搜索1次时：
- 底座EM为25%，LoRA为24%。
- 完整标准引用率分别为3%和14%。

引用覆盖改善在单次搜索条件下仍存在。
答案收益仅在本次允许追加搜索的比较中出现；
尚不能排除更多实际搜索和运行完成率改善的贡献。

## 环境

- Python 3.11，Windows。
- RTX 5060 8GB。
- PyTorch 2.9.1，CUDA 12.8构建。
- Transformers 4.57.1。
- 依赖见 requirements-lock.txt。
- 模型与固定revision见 configs/model.json。

当前多个脚本将HF_HOME固定为：
D:\search-agent-cache\huggingface

换电脑时需检查相关脚本的缓存路径。
依赖锁定基于本机环境，不保证跨系统完全一致。

## 数据

- data/synthetic_dev_v1：合成开发与控制器调试数据。
- data/hotpot_bridge_v1：100题开发集及固定文档库。
- data/hotpot_bridge_v1/smoke：30题开发子集。
- data/hotpot_bridge_v1/test：200题保留测试。
- data/hotpot_train_v1：300道训练题及训练语料。
- data/sft_pilot_v1：此前5条轨迹的小规模pilot。
- data/sft_reviewed_v1：本轮41条审核轨迹及数据说明。

本轮训练数据包含32条单次搜索、9条多次搜索轨迹。
审核属于助手基于文档的证据检查，不是人工审定。
引用修订保留来源记录，原始轨迹未修改。

数据准备按标准证据标题等规则隔离；
不宣称完全无泄漏，也不能排除公开数据的预训练污染。

## 基线运行

以下命令从项目根目录执行。

100题开发集三种底座基线：

```powershell
python .\src\run_baselines.py --mode all --language en --data-dir ".\data\hotpot_bridge_v1"
```

默认搜索上限3次、动作上限8次、总输出上限768 token，
上下文上限4096 token、检索top_k=3。
关闭thinking，使用贪心解码。

RAG检索一次；Agent由模型决定查询与停止。
两跳关系不一定需要两次搜索。

## SFT数据与训练

本轮审核数据已导出为train.jsonl。
导出脚本依赖本地300题原始轨迹和七批审核记录；
仅有代码无法重新生成相同审核数据。

已有数据检查通过时，无需重复运行。
脚本遇到已有输出会拒绝覆盖。

```powershell
python .\scripts\prepare_sft_tokens.py --data-dir ".\data\sft_reviewed_v1"
python .\scripts\check_sft_batch.py --data-dir ".\data\sft_reviewed_v1"
python .\scripts\train_sft_reviewed.py --epochs 1
```

只监督当前助手JSON动作和结束标记。
问题、历史消息和工具结果不参与loss。

LoRA：r=8，alpha=16，q_proj/v_proj。
训练一轮、batch size 1、学习率1e-4，
开启梯度检查点，从原始底座创建新适配器。

本轮本机适配器：

```text
outputs/sft_reviewed_20261013T110629Z_ff70a414/adapter
```

权重未随Git提交。重新训练后使用新输出目录。

## 加载检查与评测

将下面的adapter路径替换为实际训练输出：

```powershell
$adapter = '.\outputs\sft_reviewed_20261013T110629Z_ff70a414\adapter'
python .\scripts\check_adapter.py --adapter-dir "$adapter"
python .\src\run_baselines.py --mode agent --language en --data-dir ".\data\hotpot_bridge_v1" --adapter-dir "$adapter"
```

将runDir替换为上一条命令生成的运行目录：

```powershell
$runDir = '.\outputs\baselines_实际时间_编号'
python .\scripts\evaluate_trace.py --run-dir "$runDir" --questions "$runDir\questions.jsonl" --answer-language en
```

单次搜索预算对照：
在底座和LoRA运行命令中均加入 --max-searches 1。

原始200题测试使用：
--data-dir ".\data\hotpot_bridge_v1\test"

该测试集现已查看；后续针对其结果修改模型后，
不能再称为未见独立测试。

## 输出与指标

每次运行保存逐题轨迹、配置和数据副本。
evaluation保存逐题指标与方法汇总。

- 答案：HotpotQA规范化EM和词级F1。
- 检索及引用：标准文档ID覆盖指标。
- 成本：搜索次数、输入/输出token及耗时。
- 失败：动作超限、token超限、非法输出等。

失败题仍计入评测分母。
标准文档ID覆盖不是官方句子级证据评分，
也不等于证据语义支持度。

## 已知限制

- 成功轨迹筛选存在选择偏差，多轮样本较少。
- 更完整引用可能伴随无关引用。
- 长文档ID复制错误及错误恢复失败仍存在。
- 重复搜索增加，未证明整体推理效率提高。
- 尚未完成多随机种子训练、重复评测与全面语义引用审核。
- 当前结论仅适用于本次固定语料和数据划分。

## 程序检查

```powershell
python .\scripts\check_controller.py
python .\scripts\check_runner.py
```

这些检查使用模拟模型与工具，不代表模型能力评测。