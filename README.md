# AI-system-work-4

AI system课程实验4 分布式训练

# Chinese Classical ↔ Modern Translation using Mengzi-T5

A fine-tuned T5-based model for bidirectional translation between Chinese classical and modern texts, built on top of the [`Langboat/mengzi-t5-base`](https://huggingface.co/Langboat/mengzi-t5-base) pretrained model and trained with the [`xmj2002/Chinese_modern_classical`](https://huggingface.co/datasets/xmj2002/Chinese_modern_classical) dataset.

## 🚀 Project Overview

This project aims to bridge the gap between ancient Chinese and contemporary Chinese by fine-tuning a powerful generative language model on a curated corpus of parallel texts.

The pretrained base model, `mengzi-t5-base`, is designed for Chinese generation tasks and serves as a strong backbone for text-to-text translation between literary and vernacular Chinese.

## 📦 Model

- **Base model**: [Langboat/mengzi-t5-base](https://huggingface.co/Langboat/mengzi-t5-base)
- **Fine-tuned task**: Chinese modern ↔ classical translation (bidirectional)
- **Architecture**: T5
- **Tokenizer**: SentencePiece tokenizer pretrained with the base model

## 📊 Dataset

- **Name**: [xmj2002/Chinese_modern_classical](https://huggingface.co/datasets/xmj2002/Chinese_modern_classical)
- **Samples**: Parallel corpus of classical Chinese and modern Chinese sentence pairs
- **Format**:

  ```json
  {
    "info": "《三十六计·假痴不癫》",
    "modern": "宁肯装作无知而不采取行动，不可装作假聪明而轻易妄动。",
    "classical": "宁伪作不知不为，不伪作假知妄为。"
  }


