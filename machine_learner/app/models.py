from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential


def create_model(dim, regress=False):
    model = Sequential()
    model.add(Embedding(dim.getnnz() + 1, 64))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    print(model.summary())

    return model
