import numpy as np
from keras.models import load_model

from machine_learner.app.train import train_model
from machine_learner.app.data import generate_preprocessing

model = train_model("Art", True, False, True, 4, 64)