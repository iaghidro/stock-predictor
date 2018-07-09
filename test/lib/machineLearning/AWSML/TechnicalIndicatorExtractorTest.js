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
        "timePeriodStart": "2017-11-13T16:00:00.000Z",
        "timePeriodEnd": "2017-11-13T16:30:00.000Z",
        "time_open": "2017-11-13T16:00:01.971Z",
        "time_close": "2017-11-13T16:29:55.033Z",
        "priceOpen": 315.01,
        "priceHigh": 316.99,
        "priceLow": 315,
        "priceClose": 315.13,
        "volumeTraded": 3610.28335626,
        "tradesCount": 1648
    }, {
        "timePeriodStart": "2017-11-13T16:30:00.000Z",
        "timePeriodEnd": "2017-11-13T17:00:00.000Z",
        "time_open": "2017-11-13T16:30:01.247Z",
        "time_close": "2017-11-13T16:59:59.837Z",
        "priceOpen": 315.13,
        "priceHigh": 315.5,
        "priceLow": 313.35,
        "priceClose": 313.65,
        "volumeTraded": 2855.84055559,
        "tradesCount": 1296
    }, {
        "timePeriodStart": "2017-11-13T17:00:00.000Z",
        "timePeriodEnd": "2017-11-13T17:30:00.000Z",
        "time_open": "2017-11-13T17:00:00.694Z",
        "time_close": "2017-11-13T17:29:50.908Z",
        "priceOpen": 313.64,
        "priceHigh": 315.3,
        "priceLow": 312.8,
        "priceClose": 314.21,
        "volumeTraded": 2418.76366768,
        "tradesCount": 1826
    }, {
        "timePeriodStart": "2017-11-13T17:30:00.000Z",
        "timePeriodEnd": "2017-11-13T18:00:00.000Z",
        "time_open": "2017-11-13T17:30:01.797Z",
        "time_close": "2017-11-13T17:59:59.942Z",
        "priceOpen": 314.22,
        "priceHigh": 317.5,
        "priceLow": 314.22,
        "priceClose": 314.71,
        "volumeTraded": 2632.23944939,
        "tradesCount": 1297
    }, {
        "timePeriodStart": "2017-11-13T18:00:00.000Z",
        "timePeriodEnd": "2017-11-13T18:30:00.000Z",
        "time_open": "2017-11-13T18:00:01.964Z",
        "time_close": "2017-11-13T18:29:59.628Z",
        "priceOpen": 314.71,
        "priceHigh": 315.37,
        "priceLow": 314.28,
        "priceClose": 320.1,
        "volumeTraded": 3405.06675822,
        "tradesCount": 1251
    }, {
        "timePeriodStart": "2017-11-13T18:30:00.000Z",
        "timePeriodEnd": "2017-11-13T19:00:00.000Z",
        "time_open": "2017-11-13T18:30:00.889Z",
        "time_close": "2017-11-13T18:59:45.498Z",
        "priceOpen": 315.09,
        "priceHigh": 315.61,
        "priceLow": 314.8,
        "priceClose": 315.6,
        "volumeTraded": 2849.83035141,
        "tradesCount": 868
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
            const values = tiExtractor.getRangeValues(2, 'priceClose');
            const expectedValues = [
                315.13,
                313.65,
                314.21
            ];
            expect(values).to.deep.equal(expectedValues);
            done();
        });
        it('should get all previous fields if lookback period goes back further than beginning', function (done) {
            const values = tiExtractor.getRangeValues(4, 'priceClose');
            const expectedValues = [
                315.13,
                313.65,
                314.21
            ];
            expect(values).to.deep.equal(expectedValues);
            done();
        });
        it('should get last two fields', function (done) {
            const values = tiExtractor.getRangeValues(1, 'priceClose');
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
            const result = tiExtractor.calculateMovingAverage(2, 'priceClose');
            expect(result).to.equal(314.33);
            done();
        });
    });
    describe('getTrend', function () {
        it('should get DOWN when trend is downward', function (done) {
            tiExtractor.list[5] = {
                priceClose: 69
            };
            tiExtractor.list[6] = {
                priceClose: 59
            };
            tiExtractor.list[7] = {
                priceClose: 49
            };
            tiExtractor.list[8] = {
                priceClose: 39
            };
            tiExtractor.list[9] = {
                priceClose: 19
            };
            tiExtractor.list[10] = {
                priceClose: 10
            };
            tiExtractor.index = 10;
            const result = tiExtractor.getTrend(5);
            expect(result).to.equal("DOWN");
            done();
        });
        it('should get UP when trend is updward', function (done) {
            tiExtractor.list[0] = {
                priceClose: 4
            };
            tiExtractor.list[1] = {
                priceClose: 9
            };
            tiExtractor.list[2] = {
                priceClose: 10
            };
            tiExtractor.list[3] = {
                priceClose: 15
            };
            tiExtractor.list[4] = {
                priceClose: 20
            };
            tiExtractor.list[5] = {
                priceClose: 19
            };
            tiExtractor.list[6] = {
                priceClose: 30
            };
            tiExtractor.list[7] = {
                priceClose: 29
            };
            tiExtractor.list[8] = {
                priceClose: 39
            };
            tiExtractor.list[9] = {
                priceClose: 45
            };
            tiExtractor.list[10] = {
                priceClose: 50
            };
            tiExtractor.list[11] = {
                priceClose: 60
            };
            tiExtractor.list[12] = {
                priceClose: 65
            };
            tiExtractor.list[13] = {
                priceClose: 63
            };
            tiExtractor.index = 13;
            const result = tiExtractor.getTrend(13);
            expect(result).to.equal("UP");
            done();
        });
    });

});
