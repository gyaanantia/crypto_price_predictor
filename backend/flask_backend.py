from flask import Flask
import lstm as lstm

app = Flask(__name__)
predictor = lstm.CryptoCurrencyPricePredictor()
predictor.build_and_train_model()


@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/day')
def one_day_prediction():
    return str(predictor.predict(1))

@app.route('/week')
def one_week_prediction():
    return str(predictor.predict(7))

@app.route('/month')
def one_month_prediction():
    return str(predictor.predict(30))

if __name__ == 'main':
    app.run(debug=True)
