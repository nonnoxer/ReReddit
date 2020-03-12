import numpy as np

from machine_learner.app.data import process_data
from machine_learner.app.models import create_model


def generate_model(subreddit):
    (trainX, testX, trainY, testY) = process_data(subreddit)

    model = create_model(trainX, True)

    model.fit(trainX, trainY, validation_data=(testX, testY),
            epochs=2, batch_size=8)

    return model