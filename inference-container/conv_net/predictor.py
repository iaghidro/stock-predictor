# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
import os
import shutil
import json
import flask
import pandas as pd
import numpy as np
import torchvision

from fastai.imports import *
from fastai.transforms import *
from fastai.model import *
from fastai.dataset import *
from fastai.structured import *
from fastai.column_data import *
from stockPredictor import StockPredictor

index = 'Timestamp'
index_predict = 'timestamp'
lookahead = 15
percentIncrease = 1.002
recordsCount = 465155
test_ratio  = 0.95
train_ratio = 0.95
lr = 1e-4
dropout = 0.04
modelName = 'btcBinaryClassificationModel'
dep = 'action'
PATH = '/opt/ml/'

cat_vars = ['TimestampYear', 'TimestampMonth', 'TimestampWeek', 'TimestampDay', 'hour', 'minute', 'TimestampDayofweek',
            'TimestampDayofyear', 'TimestampIs_month_end', 'TimestampIs_month_start', 'TimestampIs_quarter_end',
            'TimestampIs_quarter_start', 'TimestampIs_year_end', 'TimestampIs_year_start']

contin_vars = ['Open', 'Close', 'High', 'Low', 'Volume', 'TimestampElapsed',
               'volume_adi', 'volume_obv', 'volume_obvm', 'volume_cmf', 'volume_fi', 'volume_em', 'volume_vpt', 'volume_nvi',
               'volatility_atr', 'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi', 'volatility_bbli',
               'volatility_kcc', 'volatility_kch', 'volatility_kcl', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',
               'volatility_dcl', 'volatility_dchi', 'volatility_dcli', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
               'trend_ema_fast', 'trend_ema_slow', 'trend_adx', 'trend_vortex_ind_pos',
               'trend_vortex_ind_neg', 'trend_vortex_diff', 'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',
               'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down', 'trend_aroon_ind',
               'momentum_rsi', 'momentum_mfi', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
               'momentum_ao', 'others_dr', 'others_dlr', 'others_cr', 'sma5', 'sma15', 'sma30', 'sma60', 'sma90', '1Open', '1High', '1Low',
               '1Close', '1Volume', '2Open', '2High', '2Low', '2Close', '2Volume', '3Open', '3High', '3Low', '3Close', '3Volume', '4Open',
               '4High', '4Low', '4Close', '4Volume', '5Open', '5High', '5Low', '5Close', '5Volume', '6Open', '6High', '6Low', '6Close',
               '6Volume', '7Open', '7High', '7Low', '7Close', '7Volume', '8Open', '8High', '8Low', '8Close', '8Volume', '9Open', '9High',
               '9Low', '9Close', '9Volume', '10Open', '10High', '10Low', '10Close', '10Volume', '11Open', '11High', '11Low', '11Close',
               '11Volume', '12Open', '12High', '12Low', '12Close', '12Volume', '13Open', '13High', '13Low', '13Close', '13Volume',
               '14Open', '14High', '14Low', '14Close', '14Volume', '15Open', '15High', '15Low', '15Close', '15Volume', '16Open',
               '16High', '16Low', '16Close', '16Volume', '17Open', '17High', '17Low', '17Close', '17Volume', '18Open', '18High',
               '18Low', '18Close', '18Volume', '19Open', '19High', '19Low', '19Close', '19Volume', '20Open', '20High', '20Low',
               '20Close', '20Volume', '21Open', '21High', '21Low', '21Close', '21Volume', '22Open', '22High', '22Low', '22Close',
               '22Volume', '23Open', '23High', '23Low', '23Close', '23Volume', '24Open', '24High', '24Low', '24Close', '24Volume',
               '25Open', '25High', '25Low', '25Close', '25Volume', '26Open', '26High', '26Low', '26Close', '26Volume', '27Open',
               '27High', '27Low', '27Close', '27Volume'
               ]

def create_model():
    print("Create predictor")
    p = StockPredictor(pd.DataFrame(), index)
    p.read_from_feather(PATH)

    print("Split validation and test sets")
    p.split_test_train(test_ratio)

    print("Create features")
    p.train = p.apply_variable_types(p.train, cat_vars, contin_vars, dep)
    p.test = p.apply_variable_types(p.test, cat_vars, contin_vars, dep)
    apply_cats(p.test, p.train)
    df, y, nas, mapper = proc_df(p.train, dep, do_scale=True)
    nas = {}
    train_size = p.get_train_size(train_ratio)
    val_idx = p.get_validation_indexes(train_size, df)

    print("Create model")
    md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype('int'), cat_flds=cat_vars, bs=64,
                                           is_reg=False, is_multi=False)
    cat_sz = [(c, len(p.train[c].cat.categories)+1) for c in cat_vars]
    emb_szs = [(c, min(50, (c+1)//2)) for _, c in cat_sz]
    cont_size = len(df.columns)-len(cat_vars)
    activations = [500, 100]
    dropout_later_layers = [0.01, 0.1]
    m = md.get_learner(emb_szs, cont_size, dropout, 2,
                       activations, dropout_later_layers, None, True)
    m.load(modelName)
    return m


def transform_predict_data(historical_json):
    # Create predictor
    historical_df = pd.read_json(historical_json, orient='list')
    p = StockPredictor(historical_df, index_predict)
    p.sample_train(100)

    # Data Cleaning
    p.set_date_as_index_unix()
    p.normalize_train('volume', 'open', 'high',
                      'low', 'close', 'weightedPrice')

    # Feature Engineering
    p.add_ta()
    p.clean_train()
    p.set_target('Close', lookahead, percentIncrease)
    p.add_date_values()

    # Create features
    p.train = p.apply_variable_types(p.train, cat_vars, contin_vars, dep)
    apply_cats(p.train, p.train)
    df, y, nas, mapper = proc_df(p.train, dep, do_scale=True)
    return df.tail(1)  # an array with last row(s)

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ClassificationService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = create_model()

        print("Finished fetching model")
        return cls.model

    @classmethod
    def predict(cls, historical_json):
        """For the input, do the predictions and return them."""
        model = cls.get_model()
        transformed = transform_predict_data(historical_json)
        raw_prediction = model.predict_array(
            transformed[cat_vars], transformed[contin_vars])
        return raw_prediction, np.argmax(raw_prediction, axis=1)


# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ClassificationService.get_model(
    ) is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on the last record of the historical data"""
    print('INVOCATIONS - BEGIN')

    body = flask.request.data
    print("body: {}".format(str(body)))

    print("make prediction")
    raw_prediction, prediction = ClassificationService.predict(body)

    print("format prediction")
    result = {'result': {'raw': {}}}
    result['result']['raw']['sell'] = str(raw_prediction[0][0])
    result['result']['raw']['buy'] = str(raw_prediction[0][1])
    result['result']['prediction'] = str(prediction[0])

    return flask.Response(response=json.dumps(result), status=200, mimetype='application/json')


# Invoke model
ClassificationService.get_model()
