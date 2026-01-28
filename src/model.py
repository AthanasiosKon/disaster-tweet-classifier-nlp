from tensorflow import keras
from tensorflow.keras import layers

def build_model(vocab_size, max_length, embedding_dim, lstm_units):
    model = keras.Sequential([
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ),
        layers.LSTM(lstm_units, dropout=0.1),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )

    return model
