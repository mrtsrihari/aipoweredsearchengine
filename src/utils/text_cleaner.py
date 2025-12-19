import re

def clean_text(text):
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()