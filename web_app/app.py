import os

from flask import Flask, Markup, redirect, render_template, request, session
from sqlalchemy import Boolean, Column, create_engine, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tensorflow_core.python.keras.api._v2.keras.models import load_model

from machine_learner.app.scraper import scrape

# Ensure env variables set up
print(os.environ["CLIENT_ID"], os.environ["CLIENT_SECRET"], os.environ["DATABASE_URL"])

app = Flask(__name__)

engine = create_engine(os.environ["DATABASE_URL"], echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

class Subreddit(Base):
    __tablename__ = 'subreddits'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(20))
    title = Column(Boolean)
    selftext = Column(Boolean)
    link = Column(Boolean)

Base.metadata.create_all(engine)

def get_model(subreddit):    
    model = load_model(os.path.join("machine_learner", "models", "{subreddit}.h5".format(subreddit=subreddit)))
    return model

# Main page
@app.route("/")
def root():
    config = open("config.txt", "r+")
    settings = config.readlines()
    subreddits = ""
    for i in settings:
        i = i.strip().split(",")
        subreddits += "<option value='{value}'>".format(value=i[0])
    return render_template("index.html", subreddits=Markup(subreddits))

@app.route("/r", methods=["POST"])
def submit():
    subreddit = request.form["subreddit"]
    return redirect("/r/{subreddit}".format(subreddit=subreddit))

@app.route("/r/<subreddit>")
def r_subreddit(subreddit):
    settings = session.query(Subreddit).filter(Subreddit.name == subreddit).first()
    if settings is not None:
        form = ""
        print(settings.title, settings.selftext, settings.link)
        if settings.title:
            form += "<input type='text' name='title' placeholder='Title'><br>"
        if settings.selftext:
            form += "<textarea name='selftext' form='form' placeholder='Text'></textarea><br>"
        if settings.link:
            form += "<input type='file' name='link'><br>"
        return render_template("r_subreddit.html", subreddit=subreddit, form=Markup(form))
    else:
        return redirect("/")

@app.route("/analyse/done", methods=["POST"])
def analyse_done():
    subreddit = request.form["subreddit"]
    settings = session.query(Subreddit).filter(Subreddit.name == subreddit).first()
    if settings.title:
        title = request.form["title"]
    if settings.selftext:
        selftext = request.form["selftext"]
    if settings.link:
        if request.file["link"] is not None:
            pass
    model = load_model(os.join("machine_learner", "models", "{subreddit}.h5".format(subreddit=subreddit)))
    # Predict
    return render_template("results.html")


@app.route("/admin")
def admin():
    subreddits = session.query(Subreddit).all()
    table = ""
    for i in subreddits:
        table += """
        <tr>
            <td>{name}</td>
            <td>{title}</td>
            <td>{selftext}</td>
            <td>{link}</td>
        </tr>""".format(name=i.name, title=i.title, selftext=i.selftext, link=i.link)
    return render_template("admin.html", table=Markup(table))

@app.route("/admin/scrape")
def admin_scrape():
    return render_template("admin_scrape.html")

@app.route("/admin/scrape/done", methods=["POST"])
def admin_scrape_done():
    name = request.form["subreddit"]
    config = {"title": False, "selftext": False, "link": False}
    for key in config:
        if key in request.form:
            config[key] = True
    subreddit = Subreddit(name=name, title=config["title"], selftext=config["selftext"], link=config["link"])
    session.add(subreddit)
    session.commit()
    scrape(os.environ["CLIENT_ID"], os.environ["CLIENT_SECRET"], "reddit_scraper", request.form["subreddit"])
    return redirect("/admin/scrape")

def run():
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)
