import web_app.backend.app
from machine_learner.app.ml import generate_model

print(generate_model("Showerthoughts"))
web_app.app.run()