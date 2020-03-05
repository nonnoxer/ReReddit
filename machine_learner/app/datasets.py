import glob
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

def load_post_attr(subreddit):
    cols = ["title", "score", "id", "subreddit", "url", "num_comments", "selftext", "created"]
    input_path = os.path.join("..", "datasets", "r_{subreddit}.csv".format(subreddit=subreddit))
    df = pd.read_csv(input_path, sep=";;;;", header=None, names=cols)
    return df

def process_post_attr(df, train_df, test_df):
    # Text processing
    title_vectorizer = CountVectorizer()
    title_vectorizer.fit(df["title"])
    title_vectorizer.transform(df["title"]).toarray()
    selftext_vectorizer = CountVectorizer()
    selftext_vectorizer.fit(df["selftext"])
    title_vectorizer.transform(df["selftext"]).toarray()

    # Date processing

process_post_attr(load_post_attr("Showerthoughts"), [], [])
