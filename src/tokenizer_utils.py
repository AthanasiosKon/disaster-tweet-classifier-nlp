from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_tokenizer(sentences, vocab_size):
    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(sentences)
    return tokenizer

def tokenize_and_pad(tokenizer, sentences, max_length):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )
    return padded, sequences
