import re

def clean_arabic_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('إ', 'ا').replace('أ', 'ا').replace('آ', 'ا').replace('ئ', 'ي')
    text = text.replace('ؤ', 'و').replace('ى', 'ي').replace('ة', 'ه')
    text = re.sub(r'(ا)+', 'ا', text)
    text = re.sub(r'(و)+', 'و', text)
    text = re.sub(r'(ي)+', 'ي', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
