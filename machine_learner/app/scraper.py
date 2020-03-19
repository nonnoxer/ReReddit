import os
import sys

import praw

def write(file_sr, post):
    try:
        file_sr.write(
                "{title};;;;{score};;;;{id};;;;{subreddit};;;;{url};;;;{num_comments};;;;{selftext};;;;{created}\r".format(
                    title=post.title.encode("unicode_escape"),
                    score=post.score,
                    id=post.id,
                    subreddit=post.subreddit,
                    url=post.url,
                    num_comments=post.num_comments,
                    selftext=post.selftext.encode("unicode_escape"),
                    created=post.created,
                )
            )
    except:
        pass

def scrape(client_id, client_secret, user_agent, subreddit):
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    file_sr = open(os.path.join( "machine_learner", "datasets", "{subreddit}.csv".format(subreddit=subreddit)), "a+")
    sr = reddit.subreddit(subreddit)
    cats = [sr.top(limit=10000), sr.hot(limit=1000), sr.random_rising(limit=1000), sr.random_rising(limit=1000), []]
    for i in range(1000):
        cats[-1].append(sr.random())
    for cat in cats:
        for post in cat:
            write(file_sr, post)
    file_sr.close()