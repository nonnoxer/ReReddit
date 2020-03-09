import glob
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def load_post_attr(subreddit):
    input_path = os.path.join(
        "..", "datasets", "r_{subreddit}.csv".format(subreddit=subreddit))
    df = pd.read_csv(input_path, sep=";;;;", header=None, names=None)
    return df


def process_post_attr(df):
    cols = ["title", "score", "id", "subreddit",
            "url", "num_comments", "selftext", "created"]
    df.columns = cols
    # Text processing
    dataset = []
    for i in df.index:
        dataset.append(CountVectorizer().fit_transform([df.loc[i, "title"]]).toarray())

    return (dataset, df["score"])
