import praw

def scrape(client_id, client_secret, user_agent, subreddit):
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    file_sr = open("r_{subreddit}.txt".format(subreddit=subreddit), "w+")
    sr = reddit.subreddit(subreddit)
    for post in sr.top(limit=10000):
        file_sr.write(
            "{title};;{score};;{id};;{subreddit};;{url};;{num_comments};;{selftext};;{created}\r".format(
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
    file_sr.close()

def read_scrape(subreddit):
    file_sr = open("r_{subreddit}.txt".format(subreddit=subreddit), "r")
    posts = []
    posts = file_sr.read().split("\r")
    for i in range(len(posts)):
        posts[i] = posts[i].split(";;")