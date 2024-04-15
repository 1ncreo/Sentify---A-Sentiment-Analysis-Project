from os import name
from keras.models import load_model
from flask import Flask, redirect, render_template, request, session
from keras.preprocessing import image
import numpy as np
from PIL import Image
import pyrebase


# Creating the app
app = Flask(__name__, static_url_path="/static")
app.secret_key = "super secret key"

# Loading the model
model = load_model("c1_lstm_model_acc_0.863.h5")

# Define the route for the home page
@app.route("/")
def home():
    # Add logic here to render the home page HTML template
    return render_template("index.html", session=session)