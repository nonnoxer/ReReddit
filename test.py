import numpy as np
from keras.models import load_model

from machine_learner.app.train import train_model
from machine_learner.app.data import generate_preprocessing

model = train_model("PrequelMemes", True, False, True, 16, 64, True)