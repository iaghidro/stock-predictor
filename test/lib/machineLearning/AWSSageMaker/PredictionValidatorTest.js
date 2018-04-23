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
    util,
    Logger
} = cryptoWallet;

var app = require('./../../../../lib');
const {
    PredictionValidator
} = app;


const trainingData = {
    "start": "2018-01-01T00:00:00",
    "target": [0.05350005, 0.05350005, 0.05349299, 0.05350005, 0.0535327, 0.0535327, 0.05374889, 0.05386177, 0.053888, 0.053848, 0.05380078]
};

describe.skip('lib::PredictionValidator', function () {
    var predictionValidator;
    var slack;
    var errorStub;
    var generatePredictionIndexesStub;

    beforeEach(function (done) {
        predictionValidator = new PredictionValidator({
            trainingData,
            numberOfPredictions: 3
        });
        slack = sinon.stub(Slack.prototype, 'postMessage');
        errorStub = sinon.stub(Logger.prototype, 'error');
        generatePredictionIndexesStub = sinon.stub(predictionValidator, 'generatePredictionIndexes');
        generatePredictionIndexesStub.resolves();
        done();
    });

    afterEach(function (done) {
        slack.restore();
        errorStub.restore();
        generatePredictionIndexesStub.restore();
        done();
    });

    describe('construct', function () {
        it('should set appropriate values', function (done) {
            const predictionValidator = new PredictionValidator({
                trainingData,
                numberOfPredictions: 5
            });
            expect(predictionValidator.trainingData).to.deep.equal(trainingData);
            expect(predictionValidator.numberOfPredictions).to.equal(5);
            expect(predictionValidator.predictionIndexes).to.deep.equal([]);
            done();
        });
    });

    describe('execute', function () {
        describe('generatePredictionIndexes', function () {
            it('should generate 3 random prediction indexes', function (done) {
                generatePredictionIndexesStub.restore();
                predictionValidator.execute()
                        .then(() => {
                            expect(predictionValidator.predictionIndexes.length).to.equal(3);
                            expect(isValidPredictionIndex(predictionValidator.predictionIndexes[0], 11)).to.equal(true);
                            expect(isValidPredictionIndex(predictionValidator.predictionIndexes[1], 11)).to.equal(true);
                            expect(isValidPredictionIndex(predictionValidator.predictionIndexes[2], 11)).to.equal(true);
                            done();
                        })
                        .catch((err) => console.error(err));
            });
        });
    });

});

function isValidPredictionIndex(predictionIndex, endIndex) {
    return !isNaN(predictionIndex) &&
            predictionIndex >= 0 &&
            predictionIndex <= endIndex;
}