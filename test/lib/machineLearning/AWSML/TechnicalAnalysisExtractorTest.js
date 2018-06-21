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
    let stockData;

    beforeEach(function (done) {
        const rawStockData = require('../../../mocks/stockData');
        sandbox = sinon.sandbox.create();
        stockData = util.clone(rawStockData);
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

});
