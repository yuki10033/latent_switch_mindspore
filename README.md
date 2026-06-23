# Latent-Switch MindSpore Dataset Pipeline

本仓库将 Latent-Switch 风格数据中的推理轨迹压缩、`latent budget`、`student sequence` 和 `supervision masks` 抽成一个独立的 MindSpore 数据加载实现。

仓库覆盖数据工程链路：从已蒸馏的 JSONL/Parquet 记录构造 SFT 样本，物化 token span 与监督掩码，并通过 `mindspore.dataset.GeneratorDataset` 输出可训练 batch。它不包含 teacher API 蒸馏、模型训练、模型评测或外部发布入口。

## 关键词

`MindSpore`、`mindspore.dataset.GeneratorDataset`、`监督掩码`、`latent budget`、`student sequence`、`teacher_kl_mask`、`推理轨迹数据工程`

## 安装

```bash
pip install -e .
```

如果需要直接创建 MindSpore Dataset，请在你的运行环境中安装 MindSpore：

```bash
pip install -e ".[mindspore]"
```

本仓库保留 `transformers` 作为 tokenizer 依赖，用于稳定识别 `<latent_think>`、`</latent_think>`、`<think>`、`</think>` 和聊天边界 token。示例命令中的 `--tokenizer` 应指向你的本地 tokenizer 路径或已配置好的 tokenizer 名称。

## 数据构造

输入应是已经完成 teacher 蒸馏的 JSONL 或 Parquet，每行至少能解析出：

- `question` / `problem` / `prompt`
- `stage1.correct_insight` 或 `correct_insight`
- `stage2.distilled_cot` 或 `assistant_cot`
- `stage2.answer` / `assistant_answer` / `ground_truth`

构造 SFT 记录：

```bash
python -m latent_switch_mindspore.cli build-sft \
  --input distilled.jsonl \
  --output data/sft_train.jsonl \
  --tokenizer PATH \
  --latent-min 1 \
  --latent-max 128
```

默认规则会用 `correct_insight` 的 token 数估计 latent budget：

```text
n_latent_steps = clamp(len(tokenize(correct_insight)) // 2, latent_min, latent_max)
```

student sequence 被渲染为：

```text
<latent_think>{latent placeholders}</latent_think><think>{compressed CoT}</think>{answer}
```

## MindSpore 加载

```python
from transformers import AutoTokenizer

from latent_switch_mindspore import create_mindspore_dataset


tokenizer = AutoTokenizer.from_pretrained("PATH", use_fast=True)
dataset = create_mindspore_dataset(
    "data/sft_train.jsonl",
    tokenizer=tokenizer,
    batch_size=2,
    max_length=4096,
    shuffle=False,
)

for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
    print(batch["input_ids"].shape)
    print(batch["teacher_kl_mask"].shape)
    break
```

输出 batch 包含：

- `input_ids`、`labels`、`loss_weights`
- `attention_mask`、`position_ids`
- `prompt_mask`、`latent_internal_mask`、`latent_boundary_mask`
- `cot_mask`、`answer_mask`、`teacher_kl_mask`
- `latent_positions`、`latent_slot_mask`
- `loss_source_positions`、`loss_target_positions`、`loss_pair_mask`
- `teacher_kl_source_positions`、`teacher_kl_target_positions`、`teacher_kl_pair_mask`

## 校验与查看

校验数据集能否稳定 materialize：

```bash
python -m latent_switch_mindspore.cli validate-dataset \
  --input data/sft_train.jsonl \
  --tokenizer PATH \
  --max-length 4096
```

查看单条样本的 span 和 mask 统计：

```bash
python -m latent_switch_mindspore.cli inspect-sample \
  --input data/sft_train.jsonl \
  --tokenizer PATH \
  --index 0
```

## 监督掩码语义

本仓库把 mask 视为数据 schema 的一部分，而不是训练代码的附属字段：

- prompt 与 assistant prefix 只作为条件，`labels` 置为 `-100`。
- latent interior placeholders 表示隐藏计算槽位，`labels` 置为 `-100`，由 `latent_internal_mask` 标出。
- `<latent_think>` 与 `</latent_think>` 是结构边界，由 `latent_boundary_mask` 标出并参与监督。
- `<think> ... </think>` 是压缩后的显式验证链，由 `cot_mask` 和可配置权重控制。
- answer tokens 是最终任务监督，由 `answer_mask` 标出；结束 token 单独加权监督。
- `teacher_kl_mask` 只覆盖显式 CoT 与答案相关位置，不覆盖 latent interior。

这对应第43章反复强调的接口：推理数据不只是文本集合，而是由结构、预算、mask 和质量报告共同组成的训练数据资产。
