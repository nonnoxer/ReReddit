import numpy as np
from sklearn.model_selection import train_test_split

import datasets
import models

df = datasets.load_post_attr("Showerthoughts")

(train, test) = train_test_split(df, test_size=0.25, random_state=42)

(trainX, trainY) = datasets.process_post_attr(train)
(testX, testY) = datasets.process_post_attr(test)

model = models.mlp(trainX, True)

model.fit(trainX, trainY, validation_data=(testX, testY),
          epochs=200, batch_size=8)