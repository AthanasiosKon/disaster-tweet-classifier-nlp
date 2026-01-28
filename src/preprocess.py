import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

STOPWORDS = set(stopwords.words("english"))

def remove_url(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def remove_punct(text):
    words = word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    return " ".join(words)

def remove_stopwords(text):
    return " ".join(
        w for w in text.split()
        if w not in STOPWORDS
    )

def clean_text(df):
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].map(remove_url)
    df["text"] = df["text"].map(remove_punct)
    df["text"] = df["text"].map(remove_stopwords)
    return df

def counter_word(text_col):
    counter = Counter()
    for text in text_col.values:
        for word in text.split():
            counter[word] += 1
    return counter
