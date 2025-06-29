from flask import Flask, render_template, request
from app.pipeline import combined_pipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            prediction = combined_pipeline(user_input)
            return render_template("result.html", prediction=prediction)
    return render_template("predict.html")
