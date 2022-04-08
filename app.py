from flask import Flask, render_template, Response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64
from lstm import CryptoCurrencyPricePredictor

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        day = int(request.form['tag'])
        model = CryptoCurrencyPricePredictor()
        model.build_and_train_model()
        price_predictions = model.predict(day)
        print(price_predictions)

        fig = Figure(figsize = (13, 7))
        ax = fig.subplots()
        ax.plot(model.hist['close'][:-day], label = "train", linewidth = 2)
        ax.plot(model.hist['close'][-day:], label = "test", linewidth = 2)
        ax.set_xlabel('time[year, month]', fontsize=10)
        ax.set_ylabel('price [USD]', fontsize=10)
        ax.legend(loc='best', fontsize=10)
        # Save it to a temporary buffer.

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        datalink = f"data:image/png;base64,{data}"
    
    else:
        day = 10
        model = CryptoCurrencyPricePredictor()
        model.build_and_train_model()
        price_predictions = model.predict(day)
        print(price_predictions)

        fig = Figure(figsize = (13, 7))
        ax = fig.subplots()
        ax.plot(model.hist['close'][:-day], label = "train", linewidth = 2)
        ax.plot(model.hist['close'][-day:], label = "test", linewidth = 2)
        ax.set_xlabel('time[year, month]', fontsize=10)
        ax.set_ylabel('price [USD]', fontsize=10)
        ax.legend(loc='best', fontsize=10)
        # Save it to a temporary buffer.

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        datalink = f"data:image/png;base64,{data}"      

    return render_template("index.html", graph_img_link = datalink, price_predictions = price_predictions)