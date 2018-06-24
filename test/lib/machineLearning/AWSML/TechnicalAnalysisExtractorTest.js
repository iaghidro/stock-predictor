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
            taExtractor.list[1].price_close = 35;
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
