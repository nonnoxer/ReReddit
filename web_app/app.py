import os

from flask import Flask, Markup, redirect, render_template, request, session
from tensorflow_core.python.keras.api._v2.keras.models import load_model

from machine_learner.app.scraper import scrape

app = Flask(__name__)

def get_model(subreddit):
    model = load_model(os.path.join("machine_learner", "models", "{subreddit}.h5".format(subreddit=subreddit)))
    return model

# Main page
@app.route("/")
def root():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    title = request.form["title"]
    selftext = request.form["selftext"]
    if "link" in request.files:
        link = request.files["link"]
    return render_template("results.html", title=title, selftext=selftext)

@app.route("/r/<subreddit>")
def r_subreddit(subreddit):
    model = get_model(subreddit)
    return render_template("r_subreddit.html")

@app.route("/admin/scrape")
def admin_scrape():
    return render_template("admin_scrape.html")

@app.route("/admin/scrape/done", methods=["POST"])
def admin_scrape_done():
    config = request.form["subreddit"]
    for i in ["title", "selftext", "link"]:
        if i in request.form:
            config += ",1"
        else:
            config += ",0"
    config_file = open("config.txt", "a+")
    config_file.write(config)

    scrape(os.environ["CLIENT_ID"], os.environ["CLIENT_SECRET"], "reddit_scraper", request.form["subreddit"])

    return redirect("/admin/scrape")

def run():
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)
