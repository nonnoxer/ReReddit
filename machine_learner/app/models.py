#from tensorflow_core._api.v2.compat.v1.keras.layers import CuDNNLSTM
from keras.engine.input_layer import Input
from keras.layers import (LSTM, Concatenate, Conv2D, Dense, Dropout, Embedding, Flatten,
                          GlobalAveragePooling1D, MaxPooling2D, concatenate)
from keras.models import Model, Sequential


def generate_model(title, selftext, link, title_words=None, selftext_words=None):
    inputs, models = [], []
    #title model
    if title:
        t_in = Input(shape=(title_words,))
        t_embed = Embedding(title_words, 16)(t_in)
        t_dense1 = Dense(16, activation="relu")(t_embed)
        t_dense2 = Dense(16, activation="relu")(t_dense1)
        t_dense3 = Dense(16, activation="relu")(t_dense2)
        t_dense4 = Dense(1, activation="relu")(t_dense3)
        inputs.append(t_in)
        models.append(t_dense4)
    #selftext model
    if selftext:
        s_in = Input(shape=(selftext_words,))
        s_embed = Embedding(selftext_words, 16)(s_in)
        s_lstm = LSTM(16)(s_embed)
        s_dense = Dense(1, activation="relu")(s_lstm)
        inputs.append(s_in)
        models.append(s_dense)
    #link model
    if link:
        l_in = Input(shape=(256, 256, 3))
        l_conv1 = Conv2D(32, (3, 3), input_shape=(3, 256, 256), activation="relu")(l_in)
        l_pool1 = MaxPooling2D(pool_size=(2, 2))(l_conv1)
        l_conv2 = Conv2D(32, (3, 3), activation="relu")(l_pool1)
        l_pool2 = MaxPooling2D(pool_size=(2, 2))(l_conv2)  
        l_flat1 = Flatten()(l_pool2)
        l_dense1 = Dense(16, activation="relu")(l_flat1)
        l_dense2 = Dense(1, activation="sigmoid")(l_dense1)
        l_flat2 = Flatten()(l_dense2)
        inputs.append(l_in)
        models.append(l_flat2)
    if len(models) > 1:
        merge = Concatenate()(models)
    else:
        merge = models[0]
    m_1 = Dense(16, activation="relu")(merge)
    m_out = Dense(1, activation="sigmoid")(m_1)
    model = Model(inputs=inputs, outputs=m_out)
    print(model.summary())
    model.compile(loss='mse', optimizer='adadelta', metrics=['mse', 'mae'])
    return model
def link_model():
    l_in = Input(shape=(256, 256, 3))
    l_1 = Conv2D(32, (3, 3), input_shape=(3, 256, 256), activation="relu")(l_in)
    l_2 = MaxPooling2D(pool_size=(2, 2))(l_1)
    l_3 = Conv2D(32, (3, 3), activation="relu")(l_2)
    l_4 = MaxPooling2D(pool_size=(2, 2))(l_3)
    l_5 = Flatten()(l_4)
    l_6 = Dense(16, activation="relu")(l_5)
    l_7 = Dense(1, activation="sigmoid")(l_6)
    model = Model(inputs=l_in, outputs=l_7)
    model.compile(loss='mse', optimizer='adadelta', metrics=['mse', 'mae'])

    return model
