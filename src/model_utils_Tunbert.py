from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

def load_model_and_tokenizer(model_name="AhmedBou/TuniBert"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
    return tokenizer, model

def tokenize_data(tokenizer, texts, max_length=128):
    return tokenizer(list(texts), truncation=True, padding=True, max_length=max_length, return_tensors="tf")

def create_tf_dataset(encodings, labels, batch_size=16, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    if shuffle:
        ds = ds.shuffle(1000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def compile_and_train(model, train_ds, val_ds, epochs=3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs)
