# from fastai.structured import *
# from fastai.column_data import *
import numpy as np
import pandas as pd
from stockPredictor.stockPredictor import StockPredictor

complete_train_data = {
    'Close': [3, 5, 6, 6, 4],
    'Open': [5, 7, 4, 4, 8],
    'Low': [8, 5, 3, 8, 7],
    'Volume': [4, 5, 3, 8, 9],
    'High': [4, 3, 1, 8, 4],
    'Timestamp': [1325317920, 1325317980, 1325318040, 1325318100, 1325318160]
}
trainData = {
    'Close': [1, 1.006, 1.003, 1.005, 1.4],
    'Open': [1.1, 1.2, 1.4, 1.8, 2],
    'Timestamp': [1325317920, 1325317980, 1325318040, 1325318100, 1325318160]
}
trainDataHold = {
    'Close': [2, 1.006, 1.003, 1.005, 1.4],
    'Open': [1.1, 1.2, 1.4, 1.8, 2],
    'Timestamp': [1325317920, 1325317980, 1325318040, 1325318100, 1325318160]
}
baseData = {
    'Close': [1, 1.25, 1.5, 2, 2.25],
    'Open': [1.1, 1.2, 1.4, 1.8, 2],
    'Timestamp': [1325317920, 1325317980, 1325318040, 1325318100, 1325318160],
    'action':    [1, 1, 0, 1, 0],
    'predicted': [0, 1, 0, 1, 1],
}
missingData = {
    'Close': [1, np.nan, 6, 34, 6],
    'Open': [np.nan, -np.inf, 78, 56, 7]
}
cleaned_data = {
    'Close': [1, 1, 6, 34, 6],
    'Open': [78, 78, 78, 56, 7]
}
testRecordsCount = 2
trainRecordsCount = 3
index = 'Timestamp'


def create_predictor(inputData):
    df = pd.DataFrame(data=inputData)
    return StockPredictor(df, index)


def test_constructor():
    df = pd.DataFrame(data=trainData)
    predictor = create_predictor(df)
    assert predictor.df.equals(df)
    assert predictor.index == 'Timestamp'

# ///////////////////////////////
# /////// DATA CLEANING /////////
# ///////////////////////////////


def test_split_test_train():
    df = pd.DataFrame(data=trainData)
    p = create_predictor(df)
    p.train = df
    p.split_test_train(0.6)
    assert p.train.equals(df.head(3).set_index('Timestamp'))
    assert p.test.equals(df.tail(2).set_index('Timestamp'))


def test_sample_train():
    df = pd.DataFrame(data=trainData)
    predictor = create_predictor(df)
    predictor.sample_train(3)
    assert predictor.train.equals(df.tail(3))


def test_set_date_as_index_unix():
    predictor = create_predictor(pd.DataFrame(data=trainData))
    predictor.train = predictor.df
    predictor.set_date_as_index_unix()
    timeData = {
        'Timestamp': ['2011-12-31 07:52:00', '2011-12-31 07:53:00', '2011-12-31 07:54:00', '2011-12-31 07:55:00', '2011-12-31 07:56:00']
    }
    timestamps = pd.DataFrame(data=timeData)
    timestamps.Timestamp = pd.to_datetime(timestamps.Timestamp)
    assert predictor.train.Timestamp.equals(timestamps.Timestamp)


def test_clean_train():
    predictor = create_predictor(trainData)
    predictor.train = pd.DataFrame(data=missingData)
    predictor.clean_train()
    expectedDf = pd.DataFrame(data=cleaned_data)
    expectedDf.Close = expectedDf.Close.astype(float)
    expectedDf.Open = expectedDf.Open.astype(float)
    assert predictor.train.equals(expectedDf)


def test_trim_ends():
    p = create_predictor(trainData)
    p.train = p.df
    expectedDf = p.df.tail(4).head(2)
    p.trim_ends(1, 2)
    assert p.train.copy().equals(expectedDf.copy())

# ///////////////////////////////
# ////// FEATURE CREATION ///////
# ///////////////////////////////


def test_get_train_size():
    df = pd.DataFrame(data=trainData)
    p = create_predictor(df)
    p.train = df
    train_size = p.get_train_size(0.6)
    assert train_size == 3


def test_get_validation_indexes():
    df = pd.DataFrame(data=trainData)
    p = create_predictor(df)
    validation_indexes = p.get_validation_indexes(3, df)
    assert len(validation_indexes) == 2

# ///////////////////////////////
# //// FEATURE ENGINEERING //////
# ///////////////////////////////


def test_get_max_lookahead():
    p = create_predictor(trainData)
    p.train = p.df
    max_df = p.get_max_lookahead(p.train, 'Close', 2)
    max_lookahead_data = {
        'expected_max': [1.006, 1.006, 1.005, 1.4, 1.4]
    }
    df = pd.DataFrame(data=max_lookahead_data)
    df['max'] = max_df
    df.max = df['max'].astype(np.float64)
    assert df.expected_max.equals(df.max)


def test_add_historical_candles():
    p = create_predictor(complete_train_data)
    p.add_historical_candles(p.df, 3)
    expected_data = {
        'Close':    [3, 5, 6, 6, 4],
        'Open':     [5, 7, 4, 4, 8],
        'Low':      [8, 5, 3, 8, 7],
        'Volume':   [4, 5, 3, 8, 9],
        'High':     [4, 3, 1, 8, 4],
        'Timestamp': [1325317920, 1325317980, 1325318040, 1325318100, 1325318160],
        '1Open':    [np.nan, 5.0,   7.0, 4.0,   4.0],
        '1High':    [np.nan, 4.0,   3.0, 1.0,   8.0],
        '1Low':     [np.nan, 8.0,   5.0, 3.0,   8.0],
        '1Close':   [np.nan, 3.0,   5.0, 6.0,   6.0],
        '1Volume':  [np.nan, 4.0,   5.0, 3.0,   8.0],
        '2Open':    [np.nan, np.nan, 5.0, 7.0,  4.0],
        '2High':    [np.nan, np.nan, 4.0, 3.0,  1.0],
        '2Low':     [np.nan, np.nan, 8.0, 5.0,  3.0],
        '2Close':   [np.nan, np.nan, 3.0, 5.0,  6.0],
        '2Volume':  [np.nan, np.nan, 4.0, 5.0,  3.0],
    }
    expected_df = pd.DataFrame(data=expected_data)
    assert p.df.sort_index().sort_index(axis=1).equals(
        expected_df.sort_index().sort_index(axis=1))


def test_get_last_lookahead():
    p = create_predictor(trainData)
    p.train = p.df
    last_df = p.get_last_lookahead(p.train, 'Close', 2)
    last_lookahead_data = {
        'expected_last': [1.003, 1.005, 1.4, np.nan, np.nan]
    }
    df = pd.DataFrame(data=last_lookahead_data)
    df['last'] = last_df
    df.last = df['last'].astype(np.float64)
    assert df.expected_last.equals(df.last)


def test_set_target():
    p = create_predictor(trainData)
    p.train = p.df
    p.set_target('Close', 2, 1.005)
    action_data = {
        'action': [1, 0, 0, 1, 0]
    }
    actions = pd.DataFrame(data=action_data)
    actions.action = actions.action.astype(int)
    assert p.train.action.equals(actions.action)


def test_set_target_historical():
    predictor = create_predictor(trainDataHold)
    predictor.train = predictor.df
    predictor.set_target_historical('Close', 2, 1.005)
    action_data = {
        'action': [0, 1, 0, 0, 0]
    }
    actions = pd.DataFrame(data=action_data)
    actions.action = actions.action.astype(int)
    assert predictor.train.action.equals(actions.action)


def test_set_target_historical_hold():
    predictor = create_predictor(trainDataHold)
    predictor.train = predictor.df
    predictor.set_target_historical_hold('Close', 2, 1.005)
    action_data = {
        'action': [0, 2, 1, 0, 0]
    }
    actions = pd.DataFrame(data=action_data)
    actions.action = actions.action.astype(int)
    assert predictor.train.action.equals(actions.action)


# ///////////////////////////////
# ///////// EVALUATION //////////
# ///////////////////////////////

def test_calculate_accuracy():
    df = pd.DataFrame(data=baseData)
    predictor = create_predictor(baseData)
    result = predictor.calculate_accuracy(df)
    expected = {
        'F1Score': .583,
        'totalAccuracy': .6,
        'buyAccuracy': .667,
        'sellAccuracy': .5,
        'totalBuyActions': 3,
        'successfulBuyPredictions': 2
    }
    assert result == expected


def test_calculate_net_profit():
    df = pd.DataFrame(data=baseData)
    predictor = create_predictor(baseData)
#    df.reset_index(inplace=True)
    predictor.calculate_net_profit(df, 100, 0)
    expected = {
        'startClose': 1,
        'endClose': 2.25,
        'startAmount': 100,
        'endAmount': 135,
        'buyAndHoldPercentIncrease': 125,
        'percentIncrease': 35,
        'percentDifference': -90,
        'totalTrades': 3
    }
    assert predictor.result == expected

# ///////////////////////////////
# /////////// UTIL //////////////
# ///////////////////////////////


#    print('******')
#    print(filledDf.to_string())
#    print('******')
#    print(expectedDf.to_string())
