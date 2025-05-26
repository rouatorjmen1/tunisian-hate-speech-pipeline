# 🇹🇳 Tunisian Dialect Hate Speech and Abuse Detection Using TunBERT
This repository provides a comprehensive pipeline for detecting hate speech, abusive language, and neutral content in the Tunisian dialect. The approach leverages the pre-trained BERT-based language model AhmedBou/TuniBert, specifically adapted for the nuances of Tunisian Arabic.

This work replicates the experimental methodology detailed in the peer-reviewed publication:

"Tunisian Dialect Hate Speech and Abuse Detection with BERT-Based Models"
Roua Torjmen and Kais Haddar —  Accepted Subject to Revision at KES 2025

##  Overview
Overview
The repository offers tools and scripts to preprocess dialectal Arabic text, fine-tune the TunBERT model on the publicly available T-HSAB dataset, and evaluate classification performance on hate speech and abusive language detection tasks.

## Key Features
Utilizes the publicly available T-HSAB dataset annotated for hate speech, abuse, and neutral content.

Implements preprocessing and normalization techniques tailored to dialectal Arabic.

Employs Hugging Face’s AutoTokenizer for tokenization aligned with the AraBERT, ARBERT, MARBERT, and TunBERT models.

Fine-tunes the transformer model using TensorFlow.

Provides evaluation metrics: F1-score, Precision, Recall, and Confusion Matrix.

Compatible with Google Colab and local execution environments.

## Repository Structure:
.

├── pipeline.ipynb                  # Main notebook implementing the full pipeline

├── requirements.txt                # Python dependencies

├── data/

│   └── sample_thsab.csv            # extract of T-HSAB dataset 

├── models/
    
│   └── tunbert_sa_tf
        
    │   ├── tuned_model.ckpt.*

    │   ├── config.json

    │   └── vocab.txt



├── src/

│   ├── preprocessing.py             # Text cleaning and normalization

│   ├── data_utils.py                # Data loading and encoding

│   ├── model_utils.py               # Model loading, compilation, and training

│   └── evaluation.py                # Evaluation, classification report, and heatmap

└── README.md                        # This document

## Dataset
This project utilizes the T-HSAB dataset: a Tunisian dialect hate speech and abusive language dataset. You can obtain the original dataset from the official repository:

👉 https://github.com/Hala-Mulki/T-HSAB-A-Tunisian-Hate-Speech-and-Abusive-Dataset

After downloading, please follow these instructions:

1. Place the dataset file (thsab.csv) into the project's data/ directory.

2. The dataset must include at least the following two columns:

      text: the raw text in Tunisian Arabic
      
      label: the class label (hate, neutral, or abuse).

3. Label Encoding:
For training and evaluation purposes, the categorie column should be mapped to numerical values as follows:

      hate → 0
      
      neutral → 1
      
      abuse → 2

🔍 You can refer to data/sample_thsab.csv provided in the repository as a format reference for preprocessing and label encoding.

Please see sample_thsab.csv like a guide

## Installation and Usage
### Clone the repository


git clone https://github.com/tunisian-hate-speech-pipeline.git
cd tunisian-hate-speech-pipeline

### Install dependencies


pip install -r requirements.txt

### Run the pipeline

Launch the Jupyter notebook locally or in Google Colab:


jupyter notebook pipeline_thsab_tunibert.ipynb
Follow the notebook’s instructions to preprocess data, fine-tune the model, and evaluate performance.

## Preprocessing
The preprocessing pipeline includes:

Arabic character normalization (e.g., إ → ا, ة → ه)

Removal of punctuation and repeated letters

Tokenization using Hugging Face’s AutoTokenizer for TunBERT

## Evaluation Metrics
The model is evaluated using:

F1-Score (macro-averaged)

Precision and Recall per class

Confusion matrix visualization


## Model

We utilize TunBERT, a transformer-based model pretrained on Tunisian dialect social media data, optimized for dialectal Arabic understanding.

To use **TunBERT-TensorFlow** for Sentiment Analysis (SA), clone or download the repository 👉 https://github.com/instadeepai/tunbert.

Create a folder: `models/tunbert_sa_tf/`

Place the following files in that folder:

- `tuned_model.ckpt.*`  
- `config.json`  
- `vocab.txt`  

### Additional Models

In addition to TunBERT, we also compare the performance of other Arabic pre-trained BERT models:

- **AraBERT**: [`aubmindlab/araBERTv02`](https://huggingface.co/aubmindlab/araBERTv02)  
- **ARABERT**: [`aubmindlab/bert-base-arabertv2`](https://huggingface.co/aubmindlab/bert-base-arabertv2)  
- **MARBERT**: [`UBC-NLP/MARBERT`](https://huggingface.co/UBC-NLP/MARBERT)  

These models are automatically downloaded from Hugging Face during pipeline execution
## License
This repository is available for academic and research use. Users are encouraged to adapt and extend the code for scholarly purposes.

## Contact
For questions, please contact:

Roua Torjmen
Email: rouatorjmen@gmail.com
