import glob
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def process_data(subreddit):
    cols = ["title", "score", "id", "subreddit",
            "url", "num_comments", "selftext", "created"]
    input_path = os.path.join(
        "machine_learner", "datasets", "{subreddit}.csv".format(subreddit=subreddit))
    df = pd.read_csv(input_path, sep=";;;;", header=None, names=cols)

    (train, test) = train_test_split(df, test_size=0.25, random_state=42)

    # Text processing
    title_vectorizer = CountVectorizer()
    title_vectorizer.fit(df["title"])
    trainX = title_vectorizer.transform(train["title"]).toarray()
    testX = title_vectorizer.transform(test["title"]).toarray()
    
    # Y processing
    score_scaler = StandardScaler()
    score_scaler.fit(df["score"].values.reshape(-1, 1))
    trainY = score_scaler.transform(train["score"].values.reshape(-1, 1))
    testY = score_scaler.transform(test["score"].values.reshape(-1, 1))

    return (trainX, testX, trainY, testY)
