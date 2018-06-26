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
    TechnicalIndicatorExtractor
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

describe('lib::AWSML::TechnicalIndicatorExtractor', function () {

//            this.timeout(100000);
    var tiExtractor;
    var slack;
    var errorStub;
    let sandbox;

    beforeEach(function (done) {
        sandbox = sinon.sandbox.create();
        tiExtractor = new TechnicalIndicatorExtractor({
            index: 2,
            list: successfulStockDataResponse
        });
        slack = sandbox.stub(Slack.prototype, 'postMessage');
        errorStub = sandbox.stub(Logger.prototype, 'error');
        done();
    });

    afterEach(function (done) {
        sandbox.restore();
        done();
    });

    describe('construct', function () {
        it('should set appropriate values', function (done) {
            const extractor = new TechnicalIndicatorExtractor({
                index: 2,
                list: successfulStockDataResponse
            });
            expect(extractor.index).to.equal(2);
            expect(extractor.list.length).to.equal(6);
            done();
        });
    });
    describe('getRangeValues', function () {
        it('should get all previous fields if lookback period goes back to beginning', function (done) {
            const values = tiExtractor.getRangeValues(2, 'price_close');
            const expectedValues = [
                315.13,
                313.65,
                314.21
            ];
            expect(values).to.deep.equal(expectedValues);
            done();
        });
        it('should get all previous fields if lookback period goes back further than beginning', function (done) {
            const values = tiExtractor.getRangeValues(4, 'price_close');
            const expectedValues = [
                315.13,
                313.65,
                314.21
            ];
            expect(values).to.deep.equal(expectedValues);
            done();
        });
        it('should get last two fields', function (done) {
            const values = tiExtractor.getRangeValues(1, 'price_close');
            const expectedValues = [
                313.65,
                314.21
            ];
            expect(values).to.deep.equal(expectedValues);
            done();
        });
    });
    describe('calculateMovingAverage', function () {
        it('should calculate average for last three numbers', function (done) {
            const result = tiExtractor.calculateMovingAverage(2, 'price_close');
            expect(result).to.equal(314.33);
            done();
        });
    });
    describe('getTrend', function () {
        it('should get DOWN when trend is downward', function (done) {
            tiExtractor.list[5] = {
                price_close: 69
            };
            tiExtractor.list[6] = {
                price_close: 59
            };
            tiExtractor.list[7] = {
                price_close: 49
            };
            tiExtractor.list[8] = {
                price_close: 39
            };
            tiExtractor.list[9] = {
                price_close: 19
            };
            tiExtractor.list[10] = {
                price_close: 10
            };
            tiExtractor.index = 10;
            const result = tiExtractor.getTrend(5);
            expect(result).to.equal("DOWN");
            done();
        });
        it('should get UP when trend is updward', function (done) {
            tiExtractor.list[0] = {
                price_close: 4
            };
            tiExtractor.list[1] = {
                price_close: 9
            };
            tiExtractor.list[2] = {
                price_close: 10
            };
            tiExtractor.list[3] = {
                price_close: 15
            };
            tiExtractor.list[4] = {
                price_close: 20
            };
            tiExtractor.list[5] = {
                price_close: 19
            };
            tiExtractor.list[6] = {
                price_close: 30
            };
            tiExtractor.list[7] = {
                price_close: 29
            };
            tiExtractor.list[8] = {
                price_close: 39
            };
            tiExtractor.list[9] = {
                price_close: 45
            };
            tiExtractor.list[10] = {
                price_close: 50
            };
            tiExtractor.list[11] = {
                price_close: 60
            };
            tiExtractor.list[12] = {
                price_close: 65
            };
            tiExtractor.list[13] = {
                price_close: 63
            };
            tiExtractor.index = 13;
            const result = tiExtractor.getTrend(13);
            expect(result).to.equal("UP");
            done();
        });
    });

});
