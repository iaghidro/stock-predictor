const moment = require('moment');
var chai = require('chai');
var sinon = require('sinon');
var sinonChai = require('sinon-chai');
var expect = chai.expect;
var assert = chai.assert;
chai.should();
chai.use(sinonChai);

var cryptoWallet = require('crypto-wallet');
const {
    constants,
    Slack,
    Logger,
    util
} = cryptoWallet;

var app = require('../../../../lib');

const {
    ModelTrainer,
    MLDataProcessor,
    config
} = app;

const successfulStockDataResponse = [{
        "time_period_start": "2017-11-13T16:00:00.000Z",
        "time_period_end": "2017-11-13T16:30:00.000Z",
        "time_open": "2017-11-13T16:00:01.971Z",
        "time_close": "2017-11-13T16:29:55.033Z",
        "price_open": 315.01,
        "price_high": 316.99,
        "price_low": 315,
        "price_close": 315.13,
        "volume_traded": 3610.28335626,
        "trades_count": 1648
    }, {
        "time_period_start": "2017-11-13T16:30:00.000Z",
        "time_period_end": "2017-11-13T17:00:00.000Z",
        "time_open": "2017-11-13T16:30:01.247Z",
        "time_close": "2017-11-13T16:59:59.837Z",
        "price_open": 315.13,
        "price_high": 315.5,
        "price_low": 313.35,
        "price_close": 313.65,
        "volume_traded": 2855.84055559,
        "trades_count": 1296
    }, {
        "time_period_start": "2017-11-13T17:00:00.000Z",
        "time_period_end": "2017-11-13T17:30:00.000Z",
        "time_open": "2017-11-13T17:00:00.694Z",
        "time_close": "2017-11-13T17:29:50.908Z",
        "price_open": 313.64,
        "price_high": 315.3,
        "price_low": 312.8,
        "price_close": 314.21,
        "volume_traded": 2418.76366768,
        "trades_count": 1826
    }, {
        "time_period_start": "2017-11-13T17:30:00.000Z",
        "time_period_end": "2017-11-13T18:00:00.000Z",
        "time_open": "2017-11-13T17:30:01.797Z",
        "time_close": "2017-11-13T17:59:59.942Z",
        "price_open": 314.22,
        "price_high": 317.5,
        "price_low": 314.22,
        "price_close": 314.71,
        "volume_traded": 2632.23944939,
        "trades_count": 1297
    }, {
        "time_period_start": "2017-11-13T18:00:00.000Z",
        "time_period_end": "2017-11-13T18:30:00.000Z",
        "time_open": "2017-11-13T18:00:01.964Z",
        "time_close": "2017-11-13T18:29:59.628Z",
        "price_open": 314.71,
        "price_high": 315.37,
        "price_low": 314.28,
        "price_close": 320.1,
        "volume_traded": 3405.06675822,
        "trades_count": 1251
    }, {
        "time_period_start": "2017-11-13T18:30:00.000Z",
        "time_period_end": "2017-11-13T19:00:00.000Z",
        "time_open": "2017-11-13T18:30:00.889Z",
        "time_close": "2017-11-13T18:59:45.498Z",
        "price_open": 315.09,
        "price_high": 315.61,
        "price_low": 314.8,
        "price_close": 315.6,
        "volume_traded": 2849.83035141,
        "trades_count": 868
    }];

describe('lib::AWSML::StockDataProcessor', function () {

//            this.timeout(100000);
    var stockDataProcessor;
    var slack;
    var errorStub;
    let sandbox;

    beforeEach(function (done) {
        sandbox = sinon.sandbox.create();
        stockDataProcessor = new MLDataProcessor({
            amountChangePercentageThreshold: 0.5,
            timeDifferenceInMinutes: 1,
            propertyFilter: config.propertyFilters.AWSML
        });
        stockDataProcessor.calculationDelay = 10;
        slack = sandbox.stub(Slack.prototype, 'postMessage');
        errorStub = sandbox.stub(Logger.prototype, 'error');
        stockDataProcessor.targetLookahead = 2;
        done();
    });

    afterEach(function (done) {
        sandbox.restore();
        done();
    });

    describe('construct', function () {
        it('should set appropriate values', function (done) {
            const stockDataProcessor = new MLDataProcessor({
                amountChangePercentageThreshold: 0.05,
                timeDifferenceInMinutes: 1
            });
            stockDataProcessor.targetLookahead = 2;
            expect(stockDataProcessor.amountChangePercentageThreshold).to.equal(0.05);
            expect(stockDataProcessor.timeDifferenceInMinutes).to.equal(1);
            done();
        });
    });

    describe('process', function () {
        const expectedResult = [
            {
                "time_period_start": 1510588800,
                "action": "HOLD",
                "twentyPeriodSMA": "HIGHER",
                "twoHundredPeriodSMA": "HIGHER",
                "isBearish": false,
                "isBullish": false,
                "MACD": 0,
                "RSICategory": "LOW",
                "BBCategory": "HIGH_HIGH",
                "TDSequential": "NONE",
                "OBV": 0,
                "RSI": 0
            },
            {
                "time_period_start": 1510590600,
                "action": "BUY",
                "twentyPeriodSMA": "LOWER",
                "twoHundredPeriodSMA": "LOWER",
                "isBearish": false,
                "isBullish": false,
                "MACD": 0,
                "RSICategory": "LOW",
                "BBCategory": "HIGH_HIGH",
                "TDSequential": "NONE",
                "OBV": -2855.840556,
                "RSI": 0
            },
            {
                "time_period_start": 1510592400,
                "action": "BUY",
                "twentyPeriodSMA": "LOWER",
                "twoHundredPeriodSMA": "LOWER",
                "isBearish": false,
                "isBullish": false,
                "MACD": 0,
                "RSICategory": "LOW",
                "BBCategory": "HIGH_HIGH",
                "TDSequential": "NONE",
                "OBV": -437.076888,
                "RSI": 0
            },
            {
                "time_period_start": 1510594200,
                "action": "BUY",
                "twentyPeriodSMA": "HIGHER",
                "twoHundredPeriodSMA": "HIGHER",
                "isBearish": false,
                "isBullish": false,
                "MACD": 0,
                "RSICategory": "LOW",
                "BBCategory": "HIGH_HIGH",
                "TDSequential": "NONE",
                "OBV": 2195.162561,
                "RSI": 0
            },
            {
                "time_period_start": 1510596000,
                "twentyPeriodSMA": "HIGHER",
                "twoHundredPeriodSMA": "HIGHER",
                "isBearish": false,
                "isBullish": false,
                "MACD": 0,
                "RSICategory": "LOW",
                "BBCategory": "HIGH_HIGH",
                "TDSequential": "NONE",
                "OBV": 5600.22932,
                "RSI": 0
            },
            {
                "time_period_start": 1510597800,
                "twentyPeriodSMA": "HIGHER",
                "twoHundredPeriodSMA": "HIGHER",
                "isBearish": false,
                "isBullish": true,
                "MACD": 0,
                "RSICategory": "LOW",
                "BBCategory": "HIGH_HIGH",
                "TDSequential": "NONE",
                "OBV": 2750.398968,
                "RSI": 0
            }]


        it('should process data correctly', function (done) {
            stockDataProcessor.process(successfulStockDataResponse)
                    .then((result) => {
                        console.log('**********')
                        console.log(JSON.stringify(result))
                        console.log('**********')
                        console.log(JSON.stringify(expectedResult))
                        console.log('**********')

                        expect(JSON.parse(JSON.stringify(result))).to.deep.equal(JSON.parse(JSON.stringify(expectedResult)));
                        done();
                    })
                    .catch((err) => console.error(err));
        });
    });

});
