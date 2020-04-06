import os

from flask import Flask, Markup, redirect, render_template, request, session
from flask_session import Session
import numpy as np
from sqlalchemy import Boolean, Column, create_engine, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tensorflow_core.python.keras.api._v2.keras.models import load_model

from machine_learner.app.ml import generate_model
from machine_learner.app.scraper import scrape
from machine_learner.app.data import generate_stuff

# Ensure env variables set up
print(os.environ["CLIENT_ID"], os.environ["CLIENT_SECRET"], os.environ["DATABASE_URL"])

app = Flask(__name__)
app.config["SESSION_TYPE"] = "memchached"
app.config["SECRET_KEY"] = 'secret'
session = Session(app)

engine = create_engine(os.environ["DATABASE_URL"], echo=True)
Base = declarative_base()
DB_Session = sessionmaker(bind=engine)
db_session = DB_Session()

class Subreddit(Base):
    __tablename__ = 'subreddits'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(20))
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
    settings = db_session.query(Subreddit).order_by(Subreddit.name).all()
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
    settings = db_session.query(Subreddit).filter(Subreddit.name==subreddit).first()
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
        session["message"] = "alert('Subreddit is unavailable to predict');"
        return redirect("/")

@app.route("/analyse/done", methods=["POST"])
def analyse_done():
    subreddit = request.form["subreddit"]
    settings = db_session.query(Subreddit).filter(Subreddit.name == subreddit).first()
    data = np.array([])
    df, title_vectorizer, score_scaler = generate_stuff(subreddit)
    title = ""
    selftext = ""
    if settings.title:
        title = request.form["title"]
        data = np.append(data, title_vectorizer.transform([title]).toarray())
    if settings.selftext:
        selftext = request.form["selftext"]
        data = np.append(data, selftext)
    if settings.link:
        if request.file["link"] is not None:
            pass
    model = load_model(os.path.join("machine_learner", "models", "{subreddit}.h5".format(subreddit=subreddit)))
    prediction = model.predict(data.reshape(1, -1))
    prediction = int(score_scaler.inverse_transform(prediction[0])[0])
    return render_template("results.html", title=title, selftext=selftext, prediction=prediction)


@app.route("/admin")
def admin():
    subreddits = db_session.query(Subreddit).order_by(Subreddit.id).all()
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
            <td><input type="submit" value="Update">
                <form action="admin/delete/done" method="POST">
                    <input type="text" name="title" value="{name}" readonly hidden>
                    <input type="submit" value="Delete" style="color: red">
                </form></td>
        </form></tr>""".format(name=i.name, title=temp[0], selftext=temp[1], link=temp[2])
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
    subreddit = db_session.query(Subreddit).filter(Subreddit.name==name).first()
    if subreddit is not None:
        subreddit.title, subreddit.selftext, subreddit.link = config["title"], config["selftext"], config["link"]
    db_session.commit()
    return redirect("/admin")

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
    db_session.add(subreddit)
    db_session.commit()
    scrape(os.environ["CLIENT_ID"], os.environ["CLIENT_SECRET"], "reddit_scraper", request.form["subreddit"])
    return redirect("/admin/scrape")

@app.route("/admin/train")
def admin_train():
    return render_template("admin_train.html")

@app.route("/admin/train/done", methods=["POST"])
def admin_train_done():
    name = request.form["subreddit"]
    subreddit = db_session.query(Subreddit).filter(name==name).first()
    if subreddit is not None:
        (title, selftext, link) = (subreddit.title, subreddit.selftext, subreddit.link)
    else:
        return redirect("/admin/scrape")
    generate_model(subreddit).save("machine_learner/models/{subreddit}.h5".format(subreddit=subreddit))
    return redirect("/admin/train")

if __name__ == "__main__":
    app.run(debug=True)
