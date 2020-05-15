import os

import numpy as np
from flask import Flask, Markup, redirect, render_template, request
from sqlalchemy import Boolean, Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from keras.models import load_model

from machine_learner.app.train import train_model
from machine_learner.app.data import generate_preprocessing
from machine_learner.app.scraper import scrape

# Ensure env variables set up
print(os.environ["CLIENT_ID"], os.environ["CLIENT_SECRET"], os.environ["DATABASE_URL"])

app = Flask(__name__)
error = {"error": ""}

engine = create_engine(os.environ["DATABASE_URL"], echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

class Subreddit(Base):
    __tablename__ = 'subreddits'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(20), unique=True)
    title = Column(Boolean)
    selftext = Column(Boolean)
    link = Column(Boolean)

#Base.metadata.create_all(engine)

def get_model(subreddit):    
    model = load_model(os.path.join("machine_learner", "models", "{subreddit}.h5".format(subreddit=subreddit)))
    return model

# Main page
@app.route("/")
def root():
    settings = session.query(Subreddit).order_by(Subreddit.name).all()
    subreddits = ""
    for i in settings:
        subreddits += "<option value='{value}'>".format(value=i.name)
    return render_template("index.html", subreddits=Markup(subreddits))

@app.route("/r", methods=["POST"])
def submit():
    subreddit = request.form["subreddit"]
    return redirect("/r/{subreddit}".format(subreddit=subreddit))

@app.route("/r/<subreddit>")
def r_subreddit(subreddit):
    settings = session.query(Subreddit).filter(Subreddit.name==subreddit).first()
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
        error["message"] = "alert('Subreddit is unavailable to predict');"
        return redirect("/")

@app.route("/analyse/done", methods=["POST"])
def analyse_done():
    subreddit = request.form["subreddit"]
    settings = session.query(Subreddit).filter(Subreddit.name == subreddit).first()
    x = []
    preprocessing = generate_preprocessing(subreddit, settings.title, settings.selftext, settings.link)
    title = ""
    selftext = ""
    if settings.title:
        title = request.form["title"]
        x.append(np.asarray(preprocessing["title_tokenizer"].texts_to_matrix([title])))
    if settings.selftext:
        selftext = request.form["selftext"]
        x.append(np.asarray(preprocessing["selftext_tokenizer"].texts_to_matrix([selftext])))
    if settings.link:
        if request.file["link"] is not None:
            pass
    model = load_model(os.path.join("machine_learner", "models", "{subreddit}.h5".format(subreddit=subreddit)))
    prediction = model.predict(np.array(x).reshape(1, -1))
    prediction = int(preprocessing["score_scaler"].inverse_transform(prediction[0])[0])
    return render_template("results.html", title=title, selftext=selftext, prediction=prediction)

@app.route("/admin")
def admin():
    subreddits = session.query(Subreddit).order_by(Subreddit.id).all()
    table = ""
    for i in subreddits:
        temp = []
        for j in (i.title, i.selftext, i.link):
            if j:
                temp.append("checked")
            else:
                temp.append("")
        print(i.title, i.selftext, i.link)
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
                </form>
                <form action="/admin/train" method="POST">
                    <input type="text" name="subreddit" value="{name}" readonly hidden>
                    <input type="submit" value="Train">
            </td>
        </tr>""".format(name=i.name, title=temp[0], selftext=temp[1], link=temp[2])
    return render_template("admin.html", table=Markup(table))

@app.route("/admin/edit/done", methods=["POST"])
def admin_edit_done():
    name = request.form["name"]
    print(name)
    config = {"title": False, "selftext": False, "link": False}
    for key in config:
        if key in request.form:
            config[key] = True
    print(config)
    subreddit = session.query(Subreddit).filter(Subreddit.name==name).first()
    if subreddit is not None:
        subreddit.title, subreddit.selftext, subreddit.link = config["title"], config["selftext"], config["link"]
    session.commit()
    return redirect("/admin")

@app.route("/admin/new")
def admin_new():
    return render_template("admin_new.html")

@app.route("/admin/new/done", methods=["POST"])
def admin_new_done():
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
    subreddit = request.form["subreddit"]
    return render_template("admin_scrape.html", subreddit=subreddit)

@app.route("/admin/scrape/done", methods=["POST"])
def admin_scrape_done():
    name = request.form["subreddit"]
    scrape(os.environ["CLIENT_ID"], os.environ["CLIENT_SECRET"], "reddit_scraper", name)
    return redirect("/admin")

@app.route("/admin/train", methods=["POST"])
def admin_train():
    subreddit = request.form["subreddit"]
    return render_template("admin_train.html", subreddit=subreddit)

@app.route("/admin/train/done", methods=["POST"])
def admin_train_done():
    name = request.form["subreddit"]
    epochs = request.form["epochs"]
    batch_size = request.form["batch_size"]
    subreddit = session.query(Subreddit).filter(name==name).first()
    if subreddit is not None:
        title, selftext, link = subreddit.title, subreddit.selftext, subreddit.link
    else:
        return redirect("/admin/new")
    train_model(name, title, selftext, link, epochs, batch_size)
    return redirect("/admin")

if __name__ == "__main__":
    app.run(debug=True)
