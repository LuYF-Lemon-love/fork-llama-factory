## 目录

- [目录](#目录)
- [示例](#示例)
    - [深度混合微调](#深度混合微调)
    - [LLaMA-Pro 微调](#llama-pro-微调)
    - [FSDP+QLoRA 微调](#fsdpqlora-微调)

## 示例

#### 深度混合微调

```bash
llamafactory-cli train examples/extras/mod/llama3_full_sft.yaml
```

#### LLaMA-Pro 微调

```bash
bash examples/extras/llama_pro/expand.sh
llamafactory-cli train examples/extras/llama_pro/llama3_freeze_sft.yaml
```

#### FSDP+QLoRA 微调

```bash
bash examples/extras/fsdp_qlora/train.sh
```
