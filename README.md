# Pegasus-based Text Summarization

## Overview
This project fine-tunes the **Google Pegasus model** on the **Samsum dataset** for abstractive text summarization. The model is trained using the Hugging Face Transformers library and evaluates performance using **ROUGE** and **SacreBLEU** metrics. The final model is saved and can be deployed for generating summaries of dialogues.

## Features
- Fine-tunes `google/pegasus-cnn_dailymail` on the Samsum dataset.
- Implements **data preprocessing**, **tokenization**, and **vectorization**.
- Uses **transformers.Trainer** for training and evaluation.
- Evaluates using **ROUGE** and **SacreBLEU** metrics.
- Saves the model and tokenizer for further use.
- Supports **Google Drive integration** for saving the trained model.

## Installation
```sh
pip install transformers[sentencepiece] datasets sacrebleu py7zr rouge-score evaluate accelerate
```

## Model Training Steps
1. **Setup Environment**
   - Check GPU availability with `nvidia-smi`.
   - Load required libraries (`transformers`, `datasets`, `evaluate`, etc.).

2. **Load Dataset**
   - Use `datasets.load_dataset("samsum")` to fetch the Samsum dataset.
   - Tokenize dialogues and summaries using `AutoTokenizer`.

3. **Fine-Tune Pegasus Model**
   - Initialize `google/pegasus-cnn_dailymail` model.
   - Define a `Trainer` with `TrainingArguments`.
   - Train with gradient accumulation and learning rate scheduling.

4. **Evaluate Performance**
   - Compute **ROUGE** and **SacreBLEU** scores.
   - Generate summaries and compare against ground truth.

5. **Save & Deploy Model**
   - Save the model and tokenizer locally.
   - Upload to **Google Drive** for storage and retrieval.

## Model Inference Example
```python
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("pegasus-samsum-model")
pipe = pipeline("summarization", model="pegasus-samsum-model", tokenizer=tokenizer)

dialogue = "Can you summarize this conversation? I need a quick summary."
summary = pipe(dialogue, num_beams=8, max_length=128, length_penalty=0.8)[0]["summary_text"]
print("Summary:", summary)
```

## Saving Model to Google Drive
```python
from google.colab import drive
import shutil

drive.mount('/content/drive')
shutil.make_archive('/content/drive/My Drive/pegasus-samsum-model', 'zip', 'pegasus-samsum-model')
```

## Future Improvements
- Train on **larger datasets** for better generalization.
- Implement **custom loss functions** to enhance summarization quality.
- Deploy the model using **Streamlit** or **FastAPI** for real-time summarization.

## References
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Samsum Dataset](https://huggingface.co/datasets/samsum)
- [Google Pegasus Paper](https://arxiv.org/abs/1912.08777)

