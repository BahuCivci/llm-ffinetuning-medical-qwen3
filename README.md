
# Qwen3 Fine-Tuning on Medical Reasoning Dataset (Colab-based)

This repository contains a Colab-ready notebook for fine-tuning the `Qwen3` large language model on a medical reasoning dataset using Hugging Face's ecosystem.

## 🔍 Project Summary

The goal is to fine-tune the `Qwen3-8B` model on a dataset designed to train clinical reasoning, diagnosis, and treatment planning abilities. We leverage LoRA (Low-Rank Adaptation) to make the fine-tuning process efficient on limited hardware.

## 📁 Contents

- `qwen3_finetune_colab.ipynb`: Colab-compatible notebook containing the entire pipeline
- Hugging Face login and token setup
- Dataset loading and prompt formatting
- Model quantization (4-bit) for memory efficiency
- LoRA integration with PEFT
- Pre-training and post-training inference
- Model push to Hugging Face Hub

## 🚀 Setup Instructions (Colab)

1. Install requirements:
```python
!pip install -U accelerate peft trl bitsandbytes transformers huggingface_hub
```

2. Authenticate to Hugging Face:
```python
from huggingface_hub import login
login()  # Follow the prompt
```

3. Load and quantize the model using BitsAndBytes with 4-bit NF4 config.

4. Load 2000 samples from the dataset:
```
FreedomIntelligence/medical-o1-reasoning-SFT
```

5. Fine-tune using `SFTTrainer` with LoRA and monitor VRAM usage.

6. Push the final model to Hugging Face with:
```python
model.push_to_hub("YourModelName")
tokenizer.push_to_hub("YourModelName")
```

## 💻 Hardware Requirements

- Recommended: 80GB A100 or equivalent (e.g., Colab Pro with A100 + High RAM)
- 4-bit quantization helps fit the model into 39GB VRAM

## 📌 Notes

- Be cautious with memory. Clear GPU cache before training:
```python
import gc, torch
gc.collect()
torch.cuda.empty_cache()
```

- If out-of-memory errors occur, try reducing batch size or number of samples.

---

Created using 🤗 Hugging Face, Datacamp, and Colab.
