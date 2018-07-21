const moment = require('moment');
var chai = require('chai');
var sinon = require('sinon');
var sinonChai = require('sinon-chai');
var expect = chai.expect;
var assert = chai.assert;
chai.should();
chai.use(sinonChai);

var cryptoWallet = require('crypto-wallet');
var app = require('../../../../lib');
const {
    constants,
    Slack,
    Logger,
    util
} = cryptoWallet;
const {
    TechnicalAnalysisExtractor
} = app;

describe('lib::AWSML::TechnicalAnalysisExtractor', function () {
    var taExtractor;
    var slack;
    var errorStub;
    let sandbox;
    let stockData = [{
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

    beforeEach(function (done) {
        sandbox = sinon.sandbox.create();
        stockData[1].RSI = 76;
        taExtractor = new TechnicalAnalysisExtractor({
            list: stockData,
            index: 1
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
            const taExtractor = new TechnicalAnalysisExtractor({
                list: stockData,
                index: 1
            });
            expect(taExtractor.index).to.equal(1);
            expect(taExtractor.list).to.deep.equal(stockData);
            done();
        });
    });
    describe('getRSICategory', function () {
        it('should return LOW category for RSI', function (done) {
            taExtractor.list[1].RSI = 23;
            const result = taExtractor.getRSICategory();
            expect(result).to.equal('LOW');
            done();
        });
    });
    describe('getBBCategory', function () {
        it('should return correct category', function (done) {
            taExtractor.list[1].BBLower = 20;
            taExtractor.list[1].BBMiddle = 30;
            taExtractor.list[1].BBUpper = 40;
            taExtractor.list[1].priceClose = 35;
            const result = taExtractor.getBBCategory();
            expect(result).to.equal('MID_HIGH');
            done();
        });
    });
    describe('getMACDZeroLineCross', function () {
        it('should get up if macd crosses from negative above 0', function (done) {
            taExtractor.list[0].MACD = -0.95334;
            taExtractor.list[1].MACD = 0.1345;
            const result = taExtractor.getMACDZeroLineCross();
            expect(result).to.equal('UP');
            done();
        });
        it('should get down if macd crosses from above 0 below', function (done) {
            taExtractor.list[0].MACD = 0.1134;
            taExtractor.list[1].MACD = -0.9435;
            const result = taExtractor.getMACDZeroLineCross();
            expect(result).to.equal('DOWN');
            done();
        });
        it('should get none if there is no 0 line cross', function (done) {
            taExtractor.list[0].MACD = 0.1345;
            taExtractor.list[1].MACD = 0.2234;
            const result = taExtractor.getMACDZeroLineCross();
            expect(result).to.equal('NONE');
            done();
        });
    });
    describe('getMACDSignalLineCross', function () {
        it('should get up if histogram crosses from negative above 0', function (done) {
            taExtractor.list[0].MACDHistogram = -0.95334;
            taExtractor.list[1].MACDHistogram = 0.1345;
            const result = taExtractor.getMACDSignalLineCross();
            expect(result).to.equal('UP');
            done();
        });
        it('should get down if macd histogram crosses from a positive to below 0', function (done) {
            taExtractor.list[0].MACDHistogram = 0.1134;
            taExtractor.list[1].MACDHistogram = -0.9435;
            const result = taExtractor.getMACDSignalLineCross();
            expect(result).to.equal('DOWN');
            done();
        });
        it('should get none if there is no histogram cross', function (done) {
            taExtractor.list[0].MACDHistogram = 0.1345;
            taExtractor.list[1].MACDHistogram = 0.2234;
            const result = taExtractor.getMACDSignalLineCross();
            expect(result).to.equal('NONE');
            done();
        });
    });
    describe('getTDSequential', function () {
        it('should get none if there is no signal', function (done) {
            const result = taExtractor.getTDSequential();
            expect(result).to.equal('NONE');
            done();
        });
    });

});
