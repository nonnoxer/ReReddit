#from tensorflow_core._api.v2.compat.v1.keras.layers import CuDNNLSTM
from keras.engine.input_layer import Input
from keras.layers import (
    LSTM, Concatenate, Conv2D, Dense, Embedding, Flatten, MaxPooling2D)
from keras.models import Model, Sequential


def generate_model(title, selftext, link):
    """Generate an ML model based on the content type of the subreddit"""
    inputs, models = [], []
    #title model
    if title:
        t_in = Input(shape=(2048,))
        t_embed = Embedding(2048, 16)(t_in)
        t_dense1 = Dense(16, activation="relu")(t_embed)
        t_dense2 = Dense(16, activation="relu")(t_dense1)
        t_dense3 = Dense(16, activation="relu")(t_dense2)
        t_dense4 = Dense(1, activation="relu")(t_dense3)
        t_flat = Flatten()(t_dense4)
        inputs.append(t_in)
        models.append(t_flat)
    #selftext model
    if selftext:
        s_in = Input(shape=(2048,))
        s_embed = Embedding(2048, 16)(s_in)
        s_dense1 = Dense(16, activation="relu")(s_embed)
        s_dense2 = Dense(16, activation="relu")(s_dense1)
        s_dense3 = Dense(16, activation="relu")(s_dense2)
        s_dense4 = Dense(1, activation="relu")(s_dense3)
        s_flat = Flatten()(s_dense4)
        inputs.append(s_in)
        models.append(s_flat)
    #link model
    if link:
        l_in = Input(shape=(128, 128, 3))
        l_conv1 = Conv2D(32, (3, 3), input_shape=(3, 128, 128), activation="relu")(l_in)
        l_pool1 = MaxPooling2D(pool_size=(2, 2))(l_conv1)
        l_conv2 = Conv2D(32, (3, 3), activation="relu")(l_pool1)
        l_pool2 = MaxPooling2D(pool_size=(2, 2))(l_conv2)  
        l_flat1 = Flatten()(l_pool2)
        l_dense1 = Dense(16, activation="relu")(l_flat1)
        l_dense2 = Dense(1, activation="relu")(l_dense1)
        inputs.append(l_in)
        models.append(l_dense2)
    if len(models) > 1:
        merge = Concatenate()(models)
    else:
        merge = models[0]
    m_dense1 = Dense(16, activation="relu")(merge)
    m_out = Dense(2, activation="relu")(m_dense1)
    model = Model(inputs=inputs, outputs=m_out)
    print(model.summary())
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model
