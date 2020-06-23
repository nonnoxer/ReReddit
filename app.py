import os
from multiprocessing import Process

import cv2
import numpy as np
from flask import Flask, Markup, redirect, render_template, request, url_for
from keras import backend
from keras.models import load_model
from sqlalchemy import Boolean, Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from werkzeug.utils import secure_filename

from machine_learner.app.data import (download_link, filter_df, preprocess,
                                      process_link)
from machine_learner.app.scraper import scrape
from machine_learner.app.train import train_model

app = Flask(__name__)

# Create database session
engine = create_engine(os.environ["DATABASE_URL"], echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

class Subreddit(Base):
    """Subreddit class for database to store content types"""

    __tablename__ = 'subreddits'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(20), unique=True)
    title = Column(Boolean)
    selftext = Column(Boolean)
    link = Column(Boolean)

# Base.metadata.create_all(engine)

# Remove files stored in static/temp directory
try:
    temp_files = os.listdir(os.path.join("static", "temp"))
    for temp_file in temp_files:
        os.remove(os.path.join("static", "temp", temp_file))
except FileNotFoundError:
    os.makedirs(os.path.join("static", "temp"))
    
@app.route("/")
def root():
    """Home page"""
    settings = session.query(Subreddit).order_by(Subreddit.name).all()
    subreddits = ""
    for i in settings:
        subreddits += f"<option value='{i.name}'>"
    return render_template("index.html", subreddits=Markup(subreddits))

@app.route("/r", methods=["POST"])
def submit():
    """Redirect to specific subreddit prediction page"""
    subreddit = request.form["subreddit"]
    return redirect(f"/r/{subreddit}")

@app.route("/r/<subreddit>")
def r_subreddit(subreddit):
    """Specific subreddit page to accept user data according to subreddit content types"""
    settings = session.query(Subreddit).filter(Subreddit.name==subreddit).first()
    if settings is not None:
        form = ""
        if settings.title:
            form += "<input type='text' name='title' placeholder='Title'><br>"
        if settings.selftext:
            form += "<textarea name='selftext' form='form' placeholder='Text'></textarea><br>"
        if settings.link:
            form += "<input type='file' name='link' accept='image/x-png, image/jpeg'><br>"
        return render_template("r_subreddit.html", subreddit=subreddit, form=Markup(form))
    else:
        return redirect("/")

@app.route("/analyse/done", methods=["POST"])
def analyse_done():
    """Process user data, predict, show results page"""
    name = request.form["subreddit"]
    subreddit = session.query(Subreddit).filter(Subreddit.name == name).first()
    x = []
    filtered_df = filter_df(name, subreddit.title, subreddit.selftext, subreddit.link)
    preprocessing = preprocess(filtered_df, subreddit.title, subreddit.selftext)
    title = ""
    selftext = ""
    link_path_show = ""
    if subreddit.title:
        title = request.form["title"]
        x.append(preprocessing["title_vectorizer"].transform([title]).toarray())
    if subreddit.selftext:
        selftext = request.form["selftext"]
        x.append(preprocessing["selftext_vectorizer"].transform([selftext]).toarray())
    if subreddit.link:
        if "link" in request.files:
            img = request.files["link"]
            link_path = os.path.join("static", "temp", secure_filename(img.filename))
            link_path_show = f"temp/{secure_filename(img.filename)}"
            img.save(link_path)
            read_img = cv2.imread(link_path)
            res = cv2.resize(read_img, (64, 64))
            link = np.asarray([res]).astype("float")
            link /= 255
            x.append(link)
    model = load_model(os.path.join("machine_learner", "models", f"{name}.h5"))
    prediction = model.predict(x)
    backend.clear_session()
    if prediction[0][0] > prediction[0][1]:
        result = "failure"
    else:
        result = "success"
    return render_template("results.html", title=title, selftext=selftext, link=link_path_show, result=result)

@app.route("/admin")
def admin():
    """Admin page for managing data and machine learning models"""
    subreddits = session.query(Subreddit).order_by(Subreddit.id).all()
    table = ""
    for i in subreddits:
        temp = []
        for j in (i.title, i.selftext, i.link):
            if j:
                temp.append("checked")
            else:
                temp.append("")
        table += """
        <tr><form action="/admin/edit/done" method="POST">
            <td><input type="text" name="name" value="{name}" readonly></td>
            <td><input type="checkbox" name="title" value="title" {title}></td>
            <td><input type="checkbox" name="selftext" value="selftext" {selftext}></td>
            <td><input type="checkbox" name="link" value="link" {link}></td>
            <td><input type="submit" value="Update"></form>
                <form action="/admin/delete/done" method="POST">
                    <input type="text" name="title" value="{name}" readonly hidden>
                    <input type="submit" value="Delete" style="color: red">
                </form></td>
            <td>
                <form action="/admin/scrape" method="POST">
                    <input type="text" name="subreddit" value="{name}" readonly hidden>
                    <input type="submit" value="Scrape">
                </form>""".format(name=i.name, title=temp[0], selftext=temp[1], link=temp[2])
        if temp[2]:
            table += f"""
                <form action="/admin/download" method="POST">
                    <input type="text" name="subreddit" value="{i.name}" readonly hidden>
                    <input type="submit" value="Download">
                </form>
            """
        table += f"""
                <form action="/admin/train" method="POST">
                    <input type="text" name="subreddit" value="{i.name}" readonly hidden>
                    <input type="submit" value="Train">
                </form>
            </td>
        </tr>"""
    return render_template("admin.html", table=Markup(table))

@app.route("/admin/edit/done", methods=["POST"])
def admin_edit_done():
    """Edit content types of subreddit"""
    name = request.form["name"]
    config = {"title": False, "selftext": False, "link": False}
    for key in config:
        if key in request.form:
            config[key] = True
    subreddit = session.query(Subreddit).filter(Subreddit.name==name).first()
    if subreddit is not None:
        subreddit.title, subreddit.selftext, subreddit.link = config["title"], config["selftext"], config["link"]
    session.commit()
    return redirect("/admin")

@app.route("/admin/new")
def admin_new():
    """Page for adding a new subreddit"""
    return render_template("admin_new.html")

@app.route("/admin/new/done", methods=["POST"])
def admin_new_done():
    """Add a new subreddit"""
    name = request.form["subreddit"]
    config = {"title": False, "selftext": False, "link": False}
    for key in config:
        if key in request.form:
            config[key] = True
    subreddit = Subreddit(name=name, title=config["title"], selftext=config["selftext"], link=config["link"])
    session.add(subreddit)
    session.commit()
    return redirect("/admin")

@app.route("/admin/scrape", methods=["POST"])
def admin_scrape():
    """Page for scraping data from a subreddit"""
    subreddit = request.form["subreddit"]
    return render_template("admin_scrape.html", subreddit=subreddit)

@app.route("/admin/scrape/done", methods=["POST"])
def admin_scrape_done():
    """Scrape data from a subreddit"""
    name = request.form["subreddit"]
    scrape_process = Process(target=scrape, args=(os.environ["CLIENT_ID"], os.environ["CLIENT_SECRET"], "reddit_scraper", name))
    scrape_process.start()
    return redirect("/admin")

@app.route("/admin/download", methods=["POST"])
def admin_download():
    """Page for downloading images from a subreddit"""
    subreddit = request.form["subreddit"]
    return render_template("admin_download.html", subreddit=subreddit)

@app.route("/admin/download/done", methods=["POST"])
def admin_download_done():
    """Download images from a subreddit"""
    name = request.form["subreddit"]
    subreddit = session.query(Subreddit).filter(Subreddit.name==name).first()
    download_link_process = Process(target=download_link, args=(name, subreddit.title, subreddit.selftext, subreddit.link))
    download_link_process.start()
    return redirect("/admin")

@app.route("/admin/train", methods=["POST"])
def admin_train():
    """Page for training a subreddit machine learning model"""
    subreddit = request.form["subreddit"]
    return render_template("admin_train.html", subreddit=subreddit)

@app.route("/admin/train/done", methods=["POST"])
def admin_train_done():
    """Train a subreddit machine learning model"""
    name = request.form["subreddit"]
    epochs = int(request.form["epochs"])
    batch_size = int(request.form["batch_size"])
    subreddit = session.query(Subreddit).filter(Subreddit.name==name).first()
    if subreddit is not None:
        title, selftext, link = subreddit.title, subreddit.selftext, subreddit.link
    else:
        return redirect("/admin/new")
    train_model_process = Process(target=train_model, args=(name, title, selftext, link, epochs, batch_size))
    train_model_process.start()
    backend.clear_session()
    return redirect("/admin")

if __name__ == "__main__":
    app.run(debug=True)

