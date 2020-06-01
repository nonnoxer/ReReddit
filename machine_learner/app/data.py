import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
import praw
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def process_data(subreddit, title, selftext, link):
    """Read and filter CSV, prepare data for training"""
    filtered_df = filter_df(subreddit, title, selftext, link)
    preprocessing = preprocess(filtered_df, title, selftext)
    (train, test) = train_test_split(filtered_df, test_size=0.25, random_state=42)
    trainX, trainY = compile_data(subreddit, train, title, selftext, link, preprocessing)
    testX, testY = compile_data(subreddit, test, title, selftext, link, preprocessing)
    return (trainX, trainY, testX, testY)#, {"title_words": preprocessing.get("title_words"), "selftext_words": preprocessing.get("selftext_words")}

def filter_df(subreddit, title, selftext, link):
    """Remove bad data entries based on subreddit content type"""
    cols = ["title", "score", "id", "subreddit",
            "link", "num_comments", "selftext", "created"]
    if link:
        cols.append("ref_no")
        counter = 0
    input_path = os.path.join("machine_learner", "datasets", f"{subreddit}.csv")
    df = pd.read_csv(input_path, sep=";;;;", header=None, names=cols, engine="python")
    filtered = []
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
            if link:
                if row["link"].find(".jpg") != -1:
                    ext = ".jpg"
                elif row["link"].find(".png") != -1:
                    ext = ".png"
                row["ref_no"] = str(counter) + ext
                counter += 1
            filtered.append(row)
    filtered_df = pd.DataFrame(data=filtered)
    return filtered_df

def preprocess(df, title, selftext):
    """Generate corresponding title and selftext vectorizers"""
    preprocessing = {}
    if title:
        preprocessing["title_vectorizer"] = process_title(np.array(df["title"]))
    if selftext:
        preprocessing["selftext_vectorizer"] = process_selftext(np.array(df["selftext"]))
    return preprocessing

def compile_data(subreddit, df, title, selftext, link, preprocessing):
    """Finish preparing each part of the data based on subreddit content type"""
    x = []
    words = {}
    if title:
        x.append(preprocessing["title_vectorizer"].transform(df["title"]).toarray())
    if selftext:
        x.append(preprocessing["selftext_vectorizer"].transform(df["selftext"]).toarray())
    if link:
        link_arr = []
        link_np = process_link(subreddit, df["ref_no"]) # df["link"].to_numpy()
        for i in link_np:
            link_arr.append(i)
        link_arr = np.array(link_arr, dtype=float)
        link_arr /= 255
        x.append(link_arr)
    y = process_score(subreddit, df)
    return x, y

def process_title(title_list):
    """Generate and fit a count vectorizer on title list"""
    title_vectorizer = CountVectorizer(max_features=1024)
    title_vectorizer.fit(title_list)
    return title_vectorizer

def process_selftext(selftext_list):
    """Generate and fit a count vectorizer on selftext list"""
    selftext_vectorizer = CountVectorizer(max_features=2048)
    selftext_vectorizer.fit(selftext_list)
    return selftext_vectorizer

def process_score(subreddit, df):
    """Sort scores into success or fail based on data mean"""
    good_score = df["score"].mean()
    print(good_score)
    df = df.assign(score = df["score"] >= good_score)
    score_arr = []
    yes, no = 0, 0
    for success in df["score"]:
        if success:
            score_arr.append([0, 1])
            yes += 1
        else:
            score_arr.append([1, 0])
            no += 1
    print(f"Yes: {yes}, No: {no}")
    return np.array(score_arr)

def process_link(subreddit, ref_nos):
    """Load all corresponding images and resize them to 64x64 px"""
    res_img = []
    for i in ref_nos:
        img = cv2.imread(os.path.join("machine_learner", "link_datasets", subreddit, i))
        try:
            res = cv2.resize(img, (64, 64))
        except:
            res = [[[0, 0, 0]] * 64] * 64 #in the rare case a gif slips through, just mark it with a black picture
        res_img.append(res)
    return res_img

def download_link(subreddit, title, selftext, link):
    """Download all images for later use"""
    assert link
    if not os.path.exists(os.path.join("machine_learner", "link_datasets", subreddit)):
        os.makedirs(os.path.join("machine_learner", "link_datasets", subreddit))
    filtered_df = filter_df(subreddit, title, selftext, link)
    counter = 0
    for i in filtered_df["link"]:
        if i.find("jpg") != -1:
            ext = ".jpg"
        elif i.find("png") != -1:
            ext = ".png"
        f = open(os.path.join("machine_learner", "link_datasets", subreddit, str(counter) + ext), "wb")
        f.write(requests.get(i).content)
        f.close()
        counter += 1

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
#
