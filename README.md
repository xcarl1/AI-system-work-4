# AI-system-work-4

西安交通大学人工智能试验班大二 AI system课程实验4 分布式训练 参考代码

A fine-tuned T5-based model for bidirectional translation between Chinese classical and modern texts. The project also aims to **evaluate the effectiveness of different distributed communication backends (MPI / Gloo / NCCL)** using the [`Langboat/mengzi-t5-base`](https://huggingface.co/Langboat/mengzi-t5-base) model and the [`xmj2002/Chinese_modern_classical`](https://huggingface.co/datasets/xmj2002/Chinese_modern_classical) dataset.

---

## 🎯 Project Goal

This project has two main objectives:

1. **Translation Task**: Fine-tune a Chinese T5 model on classical ↔ modern Chinese sentence pairs.
2. **Distributed Training Evaluation**: Compare the training performance, stability, and speed of three distributed training backends:
   - MPI (`mpi4py`)
   - Gloo (PyTorch default CPU backend)
   - NCCL (NVIDIA GPU communication)

The results can help determine the most efficient communication strategy under various training hardware setups.

---

## 📦 Model

- **Base model**: [Langboat/mengzi-t5-base](https://huggingface.co/Langboat/mengzi-t5-base)
- **Architecture**: T5 (encoder-decoder)
- **Tokenizer**: SentencePiece

---

## 📊 Dataset

- **Name**: [xmj2002/Chinese_modern_classical](https://huggingface.co/datasets/xmj2002/Chinese_modern_classical)
- **Content**: Parallel modern & classical Chinese texts
- **Example**:

```json
{
  "info": "《三十六计·假痴不癫》",
  "modern": "宁肯装作无知而不采取行动，不可装作假聪明而轻易妄动。"
  "classical": "宁伪作不知不为，不伪作假知妄为。"
}



