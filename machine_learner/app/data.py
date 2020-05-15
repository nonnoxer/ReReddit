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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def process_data(subreddit, title, selftext, link):
    cols = ["title", "score", "id", "subreddit",
            "link", "num_comments", "selftext", "created"]
    input_path = os.path.join(
        "machine_learner", "datasets", "{subreddit}.csv".format(subreddit=subreddit))
    df = pd.read_csv(input_path, sep=";;;;", header=None, names=cols, engine="python")
    filtered_df = filter_df(df, title, selftext, link)
    preprocessing = preprocess(filtered_df, title, selftext)
    (train, test) = train_test_split(filtered_df, test_size=0.25, random_state=42)
    trainX, trainY = compile_data(train, title, selftext, link, preprocessing)
    testX, testY = compile_data(test, title, selftext, link, preprocessing)
    return (trainX, trainY, testX, testY)#, {"title_words": preprocessing.get("title_words"), "selftext_words": preprocessing.get("selftext_words")}

def filter_df(df, title, selftext, link):
    cols = ["title", "score", "id", "subreddit",
            "link", "num_comments", "selftext", "created"]
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
            filtered.append(row)
    filtered_df = pd.DataFrame(data=filtered)
    return filtered_df

def preprocess(df, title, selftext):
    preprocessing = {}
    if title:
        preprocessing["title_tokenizer"] = process_title(np.array(df["title"]))
        preprocessing["title_words"] = len(preprocessing["title_tokenizer"].word_index) + 1
    if selftext:
        preprocessing["selftext_tokenizer"] = process_selftext(np.array(df["selftext"]))
        preprocessing["selftext_words"] = len(preprocessing["selftext_tokenizer"].word_index) + 1
    # Link processing to be done separately
    preprocessing["score_scaler"] = process_score(np.array(df["score"]))
    return preprocessing

def process_title(title_list):
    title_tokenizer = Tokenizer(num_words=1024)
    title_tokenizer.fit_on_texts(title_list)
    return title_tokenizer

def process_selftext(selftext_list):
    selftext_tokenizer = Tokenizer(num_words=2048)
    selftext_tokenizer.fit_on_texts(selftext_list)
    return selftext_tokenizer

def process_score(score_list):
    score_scaler = StandardScaler()
    score_scaler = score_scaler.fit(score_list.reshape(-1, 1))
    return score_scaler

def compile_data(df, title, selftext, link, preprocessing):
    x = []
    words = {}
    if title:
        x.append(np.asarray(preprocessing["title_tokenizer"].texts_to_matrix(df["title"])))
        words["title_words"] = len(preprocessing["title_tokenizer"].word_index) + 1
    if selftext:
        x.append(np.asarray(preprocessing["selftext_tokenizer"].texts_to_matrix(df["selftext"])))
        words["selftext_words"] = len(preprocessing["selftext_tokenizer"].word_index) + 1
    if link:
        link_arr = []
        link_np = process_link(df["link"].to_numpy())
        for i in link_np:
            link_arr.append(i)
        link_arr = np.asarray(link_arr)
        link_arr /= 255
        x.append(link_arr)
    y = preprocessing["score_scaler"].transform(df["score"].to_numpy().reshape(-1, 1))
    return x, y

def process_link(link_list):
    counter = 0
    for i in link_list:
        if i.find("jpg") != -1:
            ext = ".jpg"
        elif i.find("png") != -1:
            ext = ".png"
        f = open(os.path.join("temp", str(counter) + ext), "wb")
        f.write(requests.get(i).content)
        f.close()
        counter += 1
    files = os.listdir("temp")
    sort_nicely(files)
    res_img = []
    for i in files:
        img = cv2.imread(os.path.join("temp", i))
        res = cv2.resize(img, (256, 256))
        res_img.append(res)
        os.remove(os.path.join("temp", i))
    return res_img

def generate_preprocessing(subreddit, title, selftext, link):
    cols = ["title", "score", "id", "subreddit",
            "link", "num_comments", "selftext", "created"]
    input_path = os.path.join(
        "machine_learner", "datasets", "{subreddit}.csv".format(subreddit=subreddit))
    df = pd.read_csv(input_path, sep=";;;;", header=None, names=cols, engine="python")
    filtered_df = filter_df(df, title, selftext, link)
    preprocessing = preprocess(filtered_df, title, selftext)
    return preprocessing

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