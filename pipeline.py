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


models_to_compare = {
    "TunBERT": "TunBERT-TF-SA",
    "AraBERT": "aubmindlab/araBERTv02",
    "ARBERT": "UBC-NLP/ARBERT",
    "MARBERT": "UBC-NLP/MARBERT"
}



results = {}

for name, model_name in models_to_compare.items():
    print(f"\n Processing model: {name}")
    
    if model_name == "local_tunbert":
        tokenizer, model = load_local_tunbert_tf(num_labels=len(label_encoder.classes_))
    else:
        tokenizer, model = load_model_and_tokenizer(model_name, num_labels=len(label_encoder.classes_))
    
    # Tokenisation
    train_encodings = tokenize_data(tokenizer, X_train)
    test_encodings = tokenize_data(tokenizer, X_test)
    
    # Datasets
    train_ds = create_tf_dataset(train_encodings, y_train)
    test_ds = create_tf_dataset(test_encodings, y_test, shuffle=False)
    
    # Entraînement
    compile_and_train(model, train_ds, test_ds)
    
    # Prédictions
    y_pred = predict_classes(model, test_ds)
    
    # Évaluation
    print(f" Results for {name}:")
    evaluate_model(y_test, y_pred, label_encoder.classes_)
    
    results[name] = {
        "y_pred": y_pred,
        "true": y_test.tolist()
    }

