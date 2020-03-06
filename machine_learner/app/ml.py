import numpy as np
from sklearn.model_selection import train_test_split

import datasets
import models

df = datasets.load_post_attr("Showerthoughts")

(train, test) = train_test_split(df, test_size=0.25, random_state=42)

trainX = datasets.process_post_attr(train)
testX = datasets.process_post_attr(test)
