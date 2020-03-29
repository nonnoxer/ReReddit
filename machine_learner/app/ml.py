from machine_learner.app.data import process_data
from machine_learner.app.models import title_model, selftext_model, link_model
import matplotlib.pyplot as plt

def generate_model(subreddit):
    (trainX, testX, trainY, testY) = process_data(subreddit)

    model = create_model(trainX, True)

    model_history = model.fit(trainX, trainY, validation_data=(testX, testY),
            epochs=1024, batch_size=64)
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return model