#from tensorflow_core._api.v2.compat.v1.keras.layers import CuDNNLSTM
from tensorflow_core.python.keras.api._v2.keras.layers import Conv2D, Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow_core.python.keras.api._v2.keras.models import Sequential


def create_model(dim, regress=False):
    model = Sequential()
    #model.add(Embedding(len(dim) + 1, 8))
    #model.add(CuDNNLSTM(8, return_sequences=True))
    #model.add(CuDNNLSTM(8))
    model.add(Dense(16, activation="tanh"))
    model.add(Dense(16, activation="tanh"))
    model.add(Dense(1, activation="relu"))
    model.compile(loss='mse', optimizer='adadelta', metrics=['mse', 'mae'])

    return model
