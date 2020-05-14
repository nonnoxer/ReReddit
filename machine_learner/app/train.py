import os

import matplotlib.pyplot as plt
import numpy as np

from machine_learner.app.data import process_data
from machine_learner.app.models import generate_model, link_model


def train_model(subreddit, title, selftext, link):
    (trainX, trainY, testX, testY), words = process_data(subreddit, title, selftext, link)
    #model = generate_model(title, selftext, link, title_words=words.get("title_words"), selftext_words=words.get("selftext_words"))
    #model_history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=1024, batch_size=64)
    print(trainX[1].shape, testX[1].shape)
    model = link_model()
    model_history = model.fit(trainX[1], trainY, validation_data=(testX[1], testY), epochs=4, batch_size=64)
    model.save(os.path.join("machine_learner", "models", f"{subreddit}.h5"))
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return model
