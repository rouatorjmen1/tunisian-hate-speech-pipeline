!pip install transformers datasets tensorflow scikit-learn matplotlib seaborn




from src.data_utils import load_and_clean_csv
from src.model_utils import (
    load_model_and_tokenizer,
    tokenize_data,
    create_tf_dataset,
    compile_and_train
)
from src.evaluation import predict_classes, evaluate_model




# Load and clean the dataset
(X_train, X_test, y_train, y_test), label_encoder = load_and_clean_csv("data/thsab.csv")






# Load TunBERT model and tokenizer
tokenizer, model = load_model_and_tokenizer()

# Tokenize the texts
train_encodings = tokenize_data(tokenizer, X_train)
test_encodings = tokenize_data(tokenizer, X_test)






# Create TensorFlow datasets
train_ds = create_tf_dataset(train_encodings, y_train)
test_ds = create_tf_dataset(test_encodings, y_test, shuffle=False)

# Train the model
history = compile_and_train(model, train_ds, test_ds)





# Predict and evaluate
y_pred = predict_classes(model, test_ds)
evaluate_model(y_test, y_pred, label_encoder.classes_)
