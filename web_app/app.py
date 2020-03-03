from flask import Flask, Markup, redirect, render_template, request, session

app = Flask(__name__)

# Main page
@app.route("/")
def root():
    return render_template("index.html")

def run():
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)