import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .preprocessing import clean_arabic_text

def load_and_clean_csv(file_path):
    data = pd.read_csv(file_path, delimiter='\t', encoding='utf-8')
    data.rename(columns={'texte': 'text', 'categorie': 'label'}, inplace=True)
    data['text'] = data['text'].astype(str).apply(clean_arabic_text)
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    return train_test_split(data['text'], data['label'], test_size=0.2, random_state=42), le
