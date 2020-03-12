import web_app.backend.app
from machine_learner.app.ml import generate_model

generate_model("Showerthoughts").save("machine_learner/models/Showerthoughts_model.h5")