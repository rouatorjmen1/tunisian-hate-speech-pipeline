from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

# Path to the TunBERT model checkpoint
model_name = "./models/tunbert_sa_tf"  # Folder containing config.json, vocab.txt, tuned_model.ckpt.*

# Number of output classes for classification (adjust according to your dataset)
num_labels = 2  # Default is 2 for binary classification. Set to 3 if you have 3 classes.

# Load tokenizer and model from the checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    from_pt=True  # Important if the model was trained using PyTorch
)

# Save the model in full TensorFlow SavedModel format
model.save_pretrained(f"{model_name}/saved_model")

print(f" Model successfully converted and saved to {model_name}/saved_model/")
