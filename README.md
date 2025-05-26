# 🇹🇳 Tunisian Dialect Hate Speech and Abuse Detection Using TunBERT
This repository provides a comprehensive pipeline for detecting hate speech, abusive language, and neutral content in the Tunisian dialect. The approach leverages the pre-trained BERT-based language model AhmedBou/TuniBert, specifically adapted for the nuances of Tunisian Arabic.

This work replicates the experimental methodology detailed in the peer-reviewed publication:

"Tunisian Dialect Hate Speech and Abuse Detection with BERT-Based Models"
Roua Torjmen and Kais Haddar — Accepted at KES 2025

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

│   └── sample_thsab.csv            # Example dataset 

├── src/

│   ├── preprocessing.py             # Text cleaning and normalization

│   ├── data_utils.py                # Data loading and encoding

│   ├── model_utils.py               # Model loading, compilation, and training

│   └── evaluation.py                # Evaluation, classification report, and heatmap

└── README.md                        # This document

## Dataset
Download the original T-HSAB dataset from the official repository:

👉 https://github.com/Hala-Mulki/T-HSAB-A-Tunisian-Hate-Speech-and-Abusive-Dataset

Place the dataset file (e.g., thsab.csv) inside the data/ directory. The file should contain two columns: texte and categorie.

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
We utilize TunBert, a transformer-based model pretrained on Tunisian dialect social media data, optimized for dialectal Arabic understanding.

## License
This repository is available for academic and research use. Users are encouraged to adapt and extend the code for scholarly purposes.

## Contact
For questions, please contact:

Roua Torjmen
Email: rouatorjmen@gmail.com
