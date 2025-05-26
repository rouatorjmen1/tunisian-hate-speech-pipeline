# 🇹🇳 Tunisian Dialect Hate Speech and Abuse Detection with TunBERT

This repository contains the full pipeline for detecting **hate speech**, **abusive language**, and **normal content** in the **Tunisian dialect** using the pre-trained BERT-based model [`AhmedBou/TuniBert`](https://huggingface.co/AhmedBou/TuniBert).

It is designed to reproduce the experiments described in the paper:

**"Tunisian Dialect Hate Speech and Abuse Detection with BERT-Based Models"**  
*Roua Torjmen and Kais Haddar – Accepted at KES 2025*

---

## 🧪 Features

- Uses the **T-HSAB dataset** (publicly available)
- Preprocessing and normalization tailored to dialectal Arabic
- Tokenization with Hugging Face Transformers
- Fine-tuning using **TensorFlow**
- Evaluation with:
  - F1-score
  - Precision & Recall
  - Confusion Matrix
- Easily executable in **Google Colab** or locally

---

## 📁 Repository Structure

.
├── pipeline_thsab_tunibert.ipynb # Main notebook with full pipeline
├── utils.py # (Optional) helper functions
├── requirements.txt # Python dependencies
├── data/
│ └── sample_thsab.csv # Example dataset (mock or real)
└── README.md



---

## 📥 Dataset

Download the original **T-HSAB** dataset from the official repository:

👉 https://github.com/Hala-Mulki/T-HSAB-A-Tunisian-Hate-Speech-and-Abusive-Dataset

Then place the file (e.g., `thsab.csv`) in the `data/` folder.  
Make sure the file contains two columns: `texte` and `categorie`.

---

## 🚀 Usage

### 1. Clone the Repository


git clone https://github.com/<your-username>/tunisian-hate-speech-pipeline.git
cd tunisian-hate-speech-pipeline


### 2. Install Dependencies

pip install -r requirements.txt

3. Launch the Notebook
Open pipeline_thsab_tunibert.ipynb in Jupyter or Google Colab, and follow the steps.

You can also run it from terminal using Jupyter:

jupyter notebook pipeline_thsab_tunibert.ipynb


## 🧼 Preprocessing Steps
Arabic normalization (إ → ا, ة → ه, etc.)

Removal of punctuation and repeated letters

Custom tokenization with AutoTokenizer

## 📊 Example Results
Once the model is trained, you will see evaluation results like this:

F1-Score: 0.97 (TunBERT)

Precision/Recall per class

Confusion Matrix:


## 🧠 Model
We use AhmedBou/TuniBert — a transformer model pretrained on social media data in the Tunisian dialect.

## 📜 License
You are free to reuse and adapt it for academic and research purposes.

🙋 Contact
For questions, please contact:

Roua Torjmen
Email: rouatorjmen@gmail.com

📌 Citation
If you use this code or dataset in your work, please cite our paper (KES 2025, paper ID: 487).


@inproceedings{torjmen2025tunisian,
  title={Tunisian Dialect Hate Speech and Abuse Detection with BERT-Based Models},
  author={Torjmen, Roua and Haddar, Kais},
  booktitle={29th International Conference on Knowledge-Based and Intelligent Information \& Engineering Systems (KES)},
  year={2025}
}
