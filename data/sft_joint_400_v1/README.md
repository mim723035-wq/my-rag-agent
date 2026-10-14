# 联合400条SFT训练数据

来源：`sft_expanded_300_v1`的300条，加上最终有效的`sft_targeted_shard_100_v2`的100条。v1标注暴露草稿不纳入。

- 400个不同父题、1070个正确动作、65,611监督token，最长2717 token。
- 两个源文件块及token样本逐字节保留，监督索引、错误动作屏蔽和语料来源没有重写。
- 38个错误动作只作上下文，不监督。源数据的控制器重放检查、哈希与token掩码均通过；合并后padding、标签保留和shift检查通过。
- 轨迹仍对应各自语料库或受控语料；联合SFT文件不是新的联合检索库。
- 原300条与新100条数据、C/D检查点不覆盖。训练脚本使用固定seed对所有动作统一打乱。

E训练命令（从项目根目录）：

```powershell
python .\scripts\train_sft_teacher_100.py --data-dir .\data\sft_joint_400_v1 --epochs 1
```

不传`--init-adapter`，从原始Qwen3-1.7B重新初始化LoRA；不是全参数预训练新基座。沿用r8/alpha16/q_proj/v_proj、lr1e-4、batch1、梯度累积1、seed42、BF16与gradient checkpointing。

与D相比，训练样本相同，但E是从底座联合打乱训练，D是先300再新增100并重置优化器。与C相比，新增数据、构成及计算量都变化，不能把结果单独归因为条数。本文件夹的数据冻结后不覆盖。


训练已完成：`outputs/sft_expanded400_20261014T051518Z_59b2bf47`，1070步；重载检查通过。当前未进行能力评测。详见 `EXPERIMENT_JOINT_E400.md`。
