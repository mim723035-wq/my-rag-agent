# 多跳搜索 Agent：基线与评测

基于 Qwen3-1.7B 和 BM25，实现不搜索、固定 RAG、多轮搜索
Agent 三种基线，并保存完整轨迹、答案、引用和运行成本。

当前阶段：V0.2 合成开发流程。
尚未进行 SFT 或 RL。

## 环境

- Python 3.11
- Windows
- NVIDIA RTX 5060 8GB
- PyTorch 2.9.1，CUDA 12.8 构建
- Transformers 4.57.1
- 其他依赖见 requirements-lock.txt
- 模型与固定 revision 见 configs/model.json

模型缓存当前固定为：
D:\search-agent-cache\huggingface

在其他电脑上运行前，需要修改 src/run_baselines.py 中的
HF_HOME，并确保对应磁盘有足够空间。

## 安装

建议创建独立环境：

```powershell
conda create -n search-agent python=3.11 pip -y
conda activate search-agent
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-lock.txt
python -m pip check
```

依赖记录基于当前 Windows 环境。
其他系统可能需要调整，不保证仅安装相同依赖即可完全复现。

## 数据

data/documents.jsonl 和 data/questions.jsonl：
最初的6篇文档、2道调试题。

data/synthetic_dev_v1/：
30篇虚构文档、12道开发题，包含：
- 单跳3题
- 两跳3题
- 干扰两跳3题
- 证据不足3题

所有合成数据仅用于开发检查。
这些问题已参与诊断，不能作为独立测试集。

## 三种基线

- no_search：不提供文档，直接回答。
- rag：用原问题检索一次，再回答。
- agent：模型生成查询，接收结果，继续搜索或结束。

两跳关系不一定需要两次搜索。
模型主动结束与程序达到上限分别记录。

## 运行

从 search-agent 项目根目录执行：

```powershell
python .\src\run_baselines.py --mode all --data-dir ".\data\synthetic_dev_v1"
```

只运行一道题：

```powershell
python .\src\run_baselines.py --mode all --data-dir ".\data\synthetic_dev_v1" --question-id syn_004_q
```

默认：
- 关闭 thinking
- 不采样
- 每题最多生成768 token，包含Agent动作
- 上下文限制4096 token
- Agent最多搜索3次、调用模型8次
- 每次搜索最多返回3篇文档

每次运行生成独立目录，保存数据副本、配置、数据哈希、
逐题轨迹和运行摘要。

## 评测

将路径替换为本次运行目录：

```powershell
$runDir = '.\outputs\baselines_实际时间_编号'
python .\scripts\evaluate_trace.py --run-dir "$runDir" --questions "$runDir\questions.jsonl"
```

evaluation目录保存：
- details.json / details.csv：逐题指标
- summary.json / summary.csv：方法汇总
- errors.json：缺失或无法读取的轨迹

模型运行失败也应计入评测，不能只统计成功完成的题。

## 程序检查

```powershell
python .\scripts\check_controller.py
python .\scripts\check_runner.py
```

当前结果：
- 8项控制器检查通过
- 5项运行循环检查通过

检查使用模拟工具与模型输出，不代表真实模型能力验证。

## 当前开发结果

运行目录：
outputs/baselines_20261012T081814Z_d0508399

| 方法 | 答案匹配 | 可回答题匹配 | 证据不足题匹配 | 平均搜索 |
|---|---|---|---|---|
| 不搜索 | 3/12 | 0/9 | 3/3 | 0 |
| RAG | 4/12 | 3/9 | 1/3 | 1 |
| Agent | 10/12 | 7/9 | 3/3 | 1.83 |

Agent正常结束11/12题。

失败案例：
- syn_004_q：查询未推进，反复引用占位符，达到动作上限。
- syn_008_q：查询未利用中间实体，未查到奖项就结束。
- 人工使用作品名和目标年份查询时，两题标准奖项文档
  均排名第一。

## 结论限制

- 仅12道合成开发题，不能证明真实任务泛化。
- 未进行训练，结果不是SFT或RL收益。
- Agent使用更多搜索和生成资源，尚未排除资源差异。
- 答案评分是规范化精确匹配，可能误判等价长句。
- 标准证据ID匹配不等于语义支持判断。
- 引用精确不代表引用完整。
- 当前重复查询指标不能覆盖文字不同但结果相同的查询。

## 后续计划

1. 引入有来源、可核验的真实问答数据。
2. 划分开发与独立测试集。
3. 完成人工抽检和错误归因。
4. 在固定预算下比较基线。
5. 构建训练轨迹，进行SFT并独立评测。