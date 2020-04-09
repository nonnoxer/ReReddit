import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_data(subreddit, title, selftext, link):
    cols = ["title", "score", "id", "subreddit",
            "link", "num_comments", "selftext", "created"]
    input_path = os.path.join(
        "machine_learner", "datasets", "{subreddit}.csv".format(subreddit=subreddit))
    df = pd.read_csv(input_path, sep=";;;;", header=None, names=cols, engine="python")
    title_list = []
    selftext_list = []
    link_list = []
    score_list = []
    for index, row in df.iterrows():
        not_bad_data = not \
            ((title and row["title"] == "b''") or \
            (selftext and row["selftext"] == "b''") or \
            (link and \
                ((row["link"].find("i.redd.it") == -1 and \
                row["link"].find("i.imgur.com") == -1) or \
                (row["link"].find(".jpg") == -1 and \
                row["link"].find(".png") == -1))))
        if not_bad_data:
            title_list.append(row["title"])
            selftext_list.append(row["selftext"])
            link_list.append(row["link"])
            score_list.append(row["score"])
    if title:
        title_vectorizer = process_title(np.array(title_list))
    if selftext:
        selftext_vectorizer = process_selftext(np.array(selftext_list))
    if link:
        link_stuff = process_link(np.array(link_list))
    score_scaler = process_score(np.array(score_list))


    '''title_vectorizer = CountVectorizer()
    title_vectorizer.fit(df["title"])

    return df, title_vectorizer, score_scaler'''

def process_title(title_list):
    title_vectorizer = CountVectorizer()
    title_vectorizer.fit(title_list)
    return title_vectorizer

def process_selftext(selftext_list):
    selftext_vectorizer = CountVectorizer()
    selftext_vectorizer.fit(selftext_list)
    return selftext_vectorizer

def process_link(link_list):
    counter = 0
    for i in link_list:
        if i.find("jpg") != -1:
            ext = ".jpg"
        elif i.find("png") != -1:
            ext = ".png"
        else:
            ext = ".gif"
        #f = open(os.path.join("temp", str(counter) + ext), "wb")
        #f.write(requests.get(i).content)
        #f.close()
        counter += 1
    files = os.listdir("temp")
    sort_nicely(files)
    res_img = []
    for i in files:
        img = cv2.imread(os.path.join("temp", i))
        res = cv2.resize(img, (256, 256))
        res_img.append(res)
    print("RESIZED", res_img[0].shape)
    return img

def process_score(score_list):
    score_scaler = StandardScaler()
    score_scaler.fit(score_list.reshape(-1, 1))
    return score_scaler

def process_data(subreddit):
    df, title_vectorizer, score_scaler = generate_stuff(subreddit)
    (train, test) = train_test_split(df, test_size=0.25, random_state=42)
    trainX = title_vectorizer.transform(train["title"]).toarray()
    testX = title_vectorizer.transform(test["title"]).toarray()
    trainY = score_scaler.transform(train["score"].values.reshape(-1, 1))
    testY = score_scaler.transform(test["score"].values.reshape(-1, 1))

    return (trainX, testX, trainY, testY)

# "Human" sorting by Ned Batchelder https://nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

clean_data("PrequelMemes", True, False, True)