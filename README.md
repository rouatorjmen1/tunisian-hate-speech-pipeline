# Tunisian Dialect Hate Speech Detection with TunBERT

This repository provides a complete pipeline for detecting hate speech and abusive language in the Tunisian dialect using BERT-based models.

## Description
We use the T-HSAB dataset and fine-tune the TuniBert' transformer model to classify user comments into:
- Hate speech
- Abusive language
- Neutral content

## 🚀 Features
- Preprocessing of Tunisian Arabic text
- Tokenization using HuggingFace Transformers
- Fine-tuning of TunBERT model with TensorFlow
- Evaluation: F1-score, precision, recall, confusion matrix

## 🧪 How to Run

### 1. Install requirements
```bash
pip install -r requirements.txt
