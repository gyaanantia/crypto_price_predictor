from flask import Flask, render_template, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import io
import base64
#
# import json
# import requests
# from keras.models import Sequential
# from keras.layers import Activation, Dense, Dropout, LSTM
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from sklearn.metrics import mean_absolute_error


from crypto_predictor import LSTM_train







app = Flask(__name__)



fig = Figure()
ax = fig.subplots()
ax.plot([1, 2])
# Save it to a temporary buffer.
buf = io.BytesIO()
fig.savefig(buf, format="png")
# Embed the result in the html output.
data = base64.b64encode(buf.getbuffer()).decode("ascii")
datalink = f"data:image/png;base64,{data}"
with open("companies/images.txt","r") as f:
    data = f.read()

@app.route('/')
def home():
    return render_template("index.html",
        graph_img_link = data
        )
