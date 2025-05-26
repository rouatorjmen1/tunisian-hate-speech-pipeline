from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def predict_classes(model, dataset):
    y_probs = tf.nn.softmax(model.predict(dataset).logits).numpy()
    return y_probs.argmax(axis=1)

def evaluate_model(y_true, y_pred, label_names):
    print(classification_report(y_true, y_pred, target_names=label_names))
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
