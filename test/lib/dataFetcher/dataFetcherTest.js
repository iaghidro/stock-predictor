var moment = require("moment");
var chai = require("chai");
var sinon = require("sinon");
var sinonChai = require("sinon-chai");
var expect = chai.expect;
var assert = chai.assert;
chai.should();
chai.use(sinonChai);

var app = require("../../../lib");

const { DataFetcher, util } = app;

describe("lib::DataFetcher", function() {
  this.timeout(5000);
  let sandbox;
  beforeEach(function(done) {
    sandbox = sinon.sandbox.create();
    done();
  });
  afterEach(function(done) {
    sandbox.restore();
    done();
  });
  describe("fetch:CoinbasePro", function() {
    it("should successfully fetch for given currency pair, exchange, and lookback", function(done) {
      const dataFetcher = new DataFetcher();
      const lookback = 5;
      dataFetcher
        .fetchSinglePair("BTC-USD", lookback, 0)
        .then(candles => {
          console.log("success");
          // console.log(candles);
          expect(candles.length).to.equal(lookback);
          expect(candles[0].Symbol).to.equal("BTC-USD");
          expect(!isNaN(candles[0].Low)).to.equal(true);
          expect(!isNaN(candles[0].High)).to.equal(true);
          expect(!isNaN(candles[0].Open)).to.equal(true);
          expect(!isNaN(candles[0].Close)).to.equal(true);
          expect(!isNaN(candles[0].Volume)).to.equal(true);
          // time checks
          const mostRecentTime = moment(candles[0].Date);
          const oldestTime = moment(candles[candles.length - 1].Date);
          const now = moment();
          expect(mostRecentTime.format("YYYY-MM-DD")).to.equal(
            now.format("YYYY-MM-DD")
          );
          const expectedOldestTime = now.subtract(lookback - 1, "days");
          expect(oldestTime.format("YYYY-MM-DD")).to.equal(
            expectedOldestTime.format("YYYY-MM-DD")
          );
          done();
        })
        .catch(err => console.log(err));
    });
    it("should successfully fetch for lookback window exactly than 300", function(done) {
      const dataFetcher = new DataFetcher();
      const lookback = 300;
      // add artificial delay because it's calling the exchange
      util
        .delay(500)
        .then(() => dataFetcher.fetchSinglePair("BTC-USD", lookback, 0))
        .then(candles => {
          console.log("success");
          // console.log(candles);
          expect(candles.length).to.equal(lookback);
          expect(candles[0].Symbol).to.equal("BTC-USD");
          expect(!isNaN(candles[0].Low)).to.equal(true);
          expect(!isNaN(candles[0].High)).to.equal(true);
          expect(!isNaN(candles[0].Open)).to.equal(true);
          expect(!isNaN(candles[0].Close)).to.equal(true);
          expect(!isNaN(candles[0].Volume)).to.equal(true);
          // time checks
          const mostRecentTime = moment(candles[0].Date);
          const oldestTime = moment(candles[candles.length - 1].Date);
          const now = moment();
          expect(mostRecentTime.format("YYYY-MM-DD")).to.equal(
            now.format("YYYY-MM-DD")
          );
          const expectedOldestTime = now.subtract(lookback - 1, "days");
          expect(oldestTime.format("YYYY-MM-DD")).to.equal(
            expectedOldestTime.format("YYYY-MM-DD")
          );
          done();
        })
        .catch(err => console.log(err));
    });
    it("should successfully fetch for lookback window larger than 300", function(done) {
      const dataFetcher = new DataFetcher();
      const lookback = 305;
      // add artificial delay because it's calling the exchange
      util
        .delay(1500)
        .then(() => dataFetcher.fetchSinglePair("BTC-USD", lookback, 0))
        .then(candles => {
          console.log("success");
          // console.log(candles);
          expect(candles.length).to.equal(lookback);
          expect(candles[0].Symbol).to.equal("BTC-USD");
          expect(!isNaN(candles[0].Low)).to.equal(true);
          expect(!isNaN(candles[0].High)).to.equal(true);
          expect(!isNaN(candles[0].Open)).to.equal(true);
          expect(!isNaN(candles[0].Close)).to.equal(true);
          expect(!isNaN(candles[0].Volume)).to.equal(true);
          // time checks
          const mostRecentTime = moment(candles[0].Date);
          const oldestTime = moment(candles[candles.length - 1].Date);
          const now = moment();
          expect(mostRecentTime.format("YYYY-MM-DD")).to.equal(
            now.format("YYYY-MM-DD")
          );
          const expectedOldestTime = now.subtract(lookback - 1, "days");
          expect(oldestTime.format("YYYY-MM-DD")).to.equal(
            expectedOldestTime.format("YYYY-MM-DD")
          );
          done();
        })
        .catch(err => console.log(err));
    });
  });
});
