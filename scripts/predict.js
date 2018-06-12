var _ = require('lodash');
var app = require('../lib');
const {
    config
} = app;

let record =    { time_period_start: 1522776600,
    price_low: 405.7,
    price_high: 407.79,
    volume_traded: 2237.01702728,
    trades_count: 620,
    price_open: 405.7,
    price_close: 407.79,
    action: 'BUY',
    tenPeriodSMA: 404.745455,
    twentyPeriodSMA: 403.18619,
    thirtyPeriodSMA: 400.612258,
    fiftyPeriodSMA: 393.678824,
    hundredPeriodSMA: 388.789703,
    twoHundredPeriodSMA: 390.426418,
    isBearish: false,
    isBullish: true,
    MACD: 4.119964,
    MACDSignal: 4.586829,
    RSI: 64.88,
    OBV: 6248.533914,
    BBUpper: 409.041443,
    BBLower: 398.751414,
    BBMiddle: 403.896429,
    isRSIBelow30: false,
    isRSIAbove70: false };

var predict = function () {
    console.log(`@@@@@@@@@ Start! @@@@@@@@@`);

    const trainer = new app.ModelTrainer();
    const predictParams = {
        MLModelId: 'ethUsdCoinbase-11ee-03d2', /* required */
        PredictEndpoint: 'https://realtime.machinelearning.us-east-1.amazonaws.com', /* required */
        Record: record
    }
    trainer.machineLearning.predict(predictParams)
            .then((response) => {
                console.log(`finished predicting model`);
                console.dir(response);
            })
            .catch((err) => console.error(err));
};

predict();