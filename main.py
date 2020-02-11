import pandas as pd
import praw

reddit = praw.Reddit(client_id='WjcoN3aHxXLwUA', client_secret='ZFOnZNObuAFuTiJw_DHkOnTOAlI', user_agent='showerthoughts_scraper')

posts = []
sr_showerthoughts = reddit.subreddit('Showerthoughts')
for post in sr_showerthoughts.top(limit=10):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
print(posts)