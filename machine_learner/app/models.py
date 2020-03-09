from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential

def mlp(dim, regress=False):
    model = Sequential()
    model.add(Embedding(len(dim) + 1, 32))
    model.add(LSTM(100))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model
